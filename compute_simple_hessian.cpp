#include "compute_simple_hessian.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "utils.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include <cstring>
#include <iomanip>

using namespace LAMMPS_NS;

ComputeSimpleHessian::ComputeSimpleHessian(LAMMPS *lmp, int narg_full, char **arg_full) :
  Compute(lmp, narg_full, arg_full),
  natomgroup(0), epsilon(0.0), iepsilon(0.0), ndofs(0),
  fglobal_new_pos(nullptr), fglobal_new_neg(nullptr), fglobal_copy(nullptr),
  hessian_matrix(nullptr),
  pair_compute_flag(0), kspace_compute_flag(0),
  write_to_file_flag(false), file_write_frequency(1) {

  if (domain->box_exist == 0)
    error->all(FLERR, "Compute simpleHessian defined before simulation box");

  int style_arg_offset = 3;
  int narg_style = narg_full - style_arg_offset;
  char **arg_style = &arg_full[style_arg_offset];

  if (narg_style < 1 || narg_style > 3) {
    error->all(FLERR, "Illegal compute simpleHessian command. Expected style arguments: epsilon [filename] [frequency]");
  }

  epsilon = utils::numeric(FLERR, arg_style[0], false, lmp);
  if (epsilon <= 0.0)
    error->all(FLERR, "Epsilon for compute simpleHessian must be positive");
  iepsilon = 1.0 / epsilon;

  if (narg_style >= 2) {
    write_to_file_flag = true;
    output_filename = arg_style[1];
  }

  if (narg_style == 3) {
    file_write_frequency = utils::inumeric(FLERR, arg_style[2], false, lmp);
    if (file_write_frequency <= 0)
      error->all(FLERR, "File write frequency for simpleHessian must be positive.");
  }

  vector_flag = 1;
  extvector = 0;

  pair_compute_flag = (force->pair && force->pair->compute_flag);
  kspace_compute_flag = (force->kspace && force->kspace->compute_flag);
}

ComputeSimpleHessian::~ComputeSimpleHessian() {
  memory->sfree(fglobal_new_pos);
  memory->sfree(fglobal_new_neg);
  memory->sfree(fglobal_copy);
  memory->sfree(hessian_matrix);

  if (hessian_file_stream.is_open()) {
    hessian_file_stream.close();
  }
}

void ComputeSimpleHessian::init() {
  natomgroup = group->count(igroup);
  ndofs = natomgroup * 3;
  size_vector = ndofs * ndofs;

  if (natomgroup == 0 && comm->me == 0) {
    error->warning(FLERR, "Group for compute simpleHessian is empty at run start (according to init).");
  }

  memory->sfree(fglobal_new_pos);
  memory->sfree(fglobal_new_neg);
  memory->sfree(fglobal_copy);
  memory->sfree(hessian_matrix);
  vector = nullptr;

  if (ndofs > 0) {
    fglobal_new_pos = (double *)memory->smalloc(ndofs * sizeof(double), "simpleHessian:fpos");
    fglobal_new_neg = (double *)memory->smalloc(ndofs * sizeof(double), "simpleHessian:fneg");
    fglobal_copy    = (double *)memory->smalloc(ndofs * sizeof(double), "simpleHessian:fcopy");
    hessian_matrix  = (double *)memory->smalloc(size_vector * sizeof(double), "simpleHessian:H");
    vector = hessian_matrix;
  } else {
    fglobal_new_pos = fglobal_new_neg = fglobal_copy = hessian_matrix = nullptr;
    vector = nullptr;
    size_vector = 0;
  }

  if (atom->tag_enable == 0)
    error->all(FLERR, "Compute simpleHessian requires atoms to have tags (atom_style with ID). Try 'atom_modify map array'.");

  if (write_to_file_flag && comm->me == 0) {
    if (update->ntimestep == update->beginstep) {
        hessian_file_stream.open(output_filename.c_str(), std::ios::out | std::ios::trunc);
    } else {
        hessian_file_stream.open(output_filename.c_str(), std::ios::out | std::ios::app);
    }
    if (!hessian_file_stream.is_open()) {
      error->one(FLERR, "Could not open Hessian output file: " + output_filename);
    }
  }

  pair_compute_flag = (force->pair && force->pair->compute_flag);
  kspace_compute_flag = (force->kspace && force->kspace->compute_flag);
}

void ComputeSimpleHessian::clear_all_atom_forces() {
  if (atom->nmax > 0 && atom->f && atom->f[0]) {
      memset(atom->f[0], 0, atom->nmax * 3 * sizeof(double));
  }
}

void ComputeSimpleHessian::compute_vector() {
  invoked_vector = update->ntimestep;

  if (natomgroup == 0) {
    if (hessian_matrix) 
      memset(hessian_matrix, 0, size_vector * sizeof(double)); 
    return;
  }

  if (!hessian_matrix && ndofs > 0) {
      error->all(FLERR, "Hessian matrix not allocated in compute_vector. This should not happen.");
      return; 
  }
  
  memset(hessian_matrix, 0, size_vector * sizeof(double));

  double **x = atom->x; 
  double **f = atom->f; 

  int eflag = 0; 
  int vflag = 0; 

  double x_displaced_store; 

  double *local_hessian_contribution = nullptr;
  if (ndofs > 0) { 
      local_hessian_contribution = (double *) memory->smalloc(size_vector * sizeof(double), "simpleHessian:localH");
      memset(local_hessian_contribution, 0, size_vector * sizeof(double));
  } else {
      return;
  }

  int global_displaced_atom_idx = -1;

  for (int i_atom_tag = 1; i_atom_tag <= atom->natoms; i_atom_tag++) {
    int local_idx_displaced = atom->map(i_atom_tag);

    if (local_idx_displaced >= 0 && (atom->mask[local_idx_displaced] & groupbit)) {
      global_displaced_atom_idx++;

      for (int j_dim = 0; j_dim < domain->dimension; j_dim++) {
        int displaced_dof_idx = global_displaced_atom_idx * 3 + j_dim;

        x_displaced_store = x[local_idx_displaced][j_dim];
        x[local_idx_displaced][j_dim] += epsilon;

        comm->forward_comm();
        clear_all_atom_forces();
        modify->pre_force(vflag);
        if (pair_compute_flag) force->pair->compute(eflag, vflag);
        if (atom->molecular) {
          if (force->bond) force->bond->compute(eflag, vflag);
          if (force->angle) force->angle->compute(eflag, vflag);
          if (force->dihedral) force->dihedral->compute(eflag, vflag);
          if (force->improper) force->improper->compute(eflag, vflag);
        }
        if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
        if (force->newton_pair || force->newton_bond) comm->reverse_comm();
        modify->post_force(vflag);

        memset(fglobal_copy, 0, ndofs * sizeof(double));
        int current_group_atom_force_idx = -1;
        for (int k_atom_tag = 1; k_atom_tag <= atom->natoms; k_atom_tag++) {
          int local_idx_k = atom->map(k_atom_tag);
          if (local_idx_k >=0 && (atom->mask[local_idx_k] & groupbit)) {
            current_group_atom_force_idx++;
            for (int l_dim = 0; l_dim < domain->dimension; l_dim++) {
              fglobal_copy[current_group_atom_force_idx * 3 + l_dim] = f[local_idx_k][l_dim];
            }
          }
        }
        MPI_Allreduce(fglobal_copy, fglobal_new_pos, ndofs, MPI_DOUBLE, MPI_SUM, world);
        x[local_idx_displaced][j_dim] = x_displaced_store;

        x[local_idx_displaced][j_dim] -= epsilon;

        comm->forward_comm();
        clear_all_atom_forces();
        modify->pre_force(vflag);
        if (pair_compute_flag) force->pair->compute(eflag, vflag);
        if (atom->molecular) {
          if (force->bond) force->bond->compute(eflag, vflag);
          if (force->angle) force->angle->compute(eflag, vflag);
          if (force->dihedral) force->dihedral->compute(eflag, vflag);
          if (force->improper) force->improper->compute(eflag, vflag);
        }
        if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
        if (force->newton_pair || force->newton_bond) comm->reverse_comm();
        modify->post_force(vflag);

        memset(fglobal_copy, 0, ndofs * sizeof(double));
        current_group_atom_force_idx = -1;
        for (int k_atom_tag = 1; k_atom_tag <= atom->natoms; k_atom_tag++) {
          int local_idx_k = atom->map(k_atom_tag);
          if (local_idx_k >=0 && (atom->mask[local_idx_k] & groupbit)) {
            current_group_atom_force_idx++;
            for (int l_dim = 0; l_dim < domain->dimension; l_dim++) {
              fglobal_copy[current_group_atom_force_idx * 3 + l_dim] = f[local_idx_k][l_dim];
            }
          }
        }
        MPI_Allreduce(fglobal_copy, fglobal_new_neg, ndofs, MPI_DOUBLE, MPI_SUM, world);
        x[local_idx_displaced][j_dim] = x_displaced_store;

        for (int force_dof_idx = 0; force_dof_idx < ndofs; force_dof_idx++) {
          double force_diff = fglobal_new_pos[force_dof_idx] - fglobal_new_neg[force_dof_idx];
          local_hessian_contribution[idx2_c(force_dof_idx, displaced_dof_idx, ndofs)] = -0.5 * force_diff * iepsilon;
        }
      } 
    } 
  } 

  if (ndofs > 0) {
    MPI_Allreduce(local_hessian_contribution, hessian_matrix, size_vector, MPI_DOUBLE, MPI_SUM, world);
    memory->sfree(local_hessian_contribution);
  }

  if (write_to_file_flag && comm->me == 0 && (update->ntimestep % file_write_frequency == 0)) {
    if (hessian_file_stream.is_open()) {
      hessian_file_stream << "Timestep: " << update->ntimestep << std::endl;
      hessian_file_stream << "natomgroup: " << natomgroup << " ndofs: " << ndofs << std::endl;
      for (int r = 0; r < ndofs; ++r) {
        for (int c = 0; c < ndofs; ++c) {
          hessian_file_stream << std::fixed << std::setw(16) << std::setprecision(8)
                              << hessian_matrix[idx2_c(r, c, ndofs)];
          if (c < ndofs - 1) hessian_file_stream << " ";
        }
        hessian_file_stream << std::endl;
      }
      hessian_file_stream << std::endl; 
      hessian_file_stream.flush(); 
    } else {
      error->one(FLERR, "Hessian file " + output_filename + " not open for writing in compute_vector");
    }
  }
}

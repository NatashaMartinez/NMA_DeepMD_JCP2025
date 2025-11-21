#ifdef COMPUTE_CLASS

ComputeStyle(simpleHessian, ComputeSimpleHessian)

#else

#ifndef LMP_COMPUTE_SIMPLEHESSIAN_H
#define LMP_COMPUTE_SIMPLEHESSIAN_H

#include "compute.h"
#include <fstream>
#include <string>

#define idx2_c(row_idx, col_idx, num_total_rows) ((col_idx * num_total_rows) + row_idx)

namespace LAMMPS_NS {

class ComputeSimpleHessian : public Compute {
 public:
  ComputeSimpleHessian(class LAMMPS *, int, char **);
  ~ComputeSimpleHessian() override;
  void init() override;
  void compute_vector() override;

 private:
  int natomgroup;
  double epsilon;
  double iepsilon;
  int ndofs;

  double *fglobal_new_pos;
  double *fglobal_new_neg;
  double *fglobal_copy;
  double *hessian_matrix;

  int pair_compute_flag;
  int kspace_compute_flag;

  void clear_all_atom_forces();

  bool write_to_file_flag;
  std::string output_filename;
  std::ofstream hessian_file_stream;
  int file_write_frequency;
};

}

#endif
#endif

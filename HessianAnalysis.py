#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eigh
from pathlib import Path
import time, math, re

HESSIAN_FILE = "solute_hessian9.dat"
TRAJ_FILE = "NewProd_5v4.lammpstrj"

PINNED_ALL_FILE = "corrected_frequencies_from_lammps_hessian9.txt"
TRACKED_ONLY_FILE = "tracked_amideI_vs_timestep.csv"

SOLUTE_ELEMENTS = ['C', 'N', 'C', 'C', 'O', 'D']
NUM_SOLUTE_ATOMS = len(SOLUTE_ELEMENTS)
IDX_C, IDX_O = 2, 4

AMIDE_I_TARGET_INDEX = 16
FREQ_CONV = 521.4706
BASE_WINDOW = (1500.0, 1700.0)
DYN_HALF_WIDTH = 1400.0

PHYS_MASS = {'H': 1.00784, 'D': 2.01410, 'C': 12.0107, 'N': 14.0067, 'O': 15.9990}
ID_MASS   = {'H': 1.00784,  'D': 2.0141,  'C': 12.0107, 'N': 14.0067, 'O': 15.9990} # YOU CAN MODIFY MASS HERE

OVERLAP_BLEND = 1.0
NH_PENALTY    = 1.0
PERP_PENALTY  = 1.0
WARN_MISMATCH_STEPS = 2

SAVE_MW_EVECS = True
MW_EVECS_DIR  = "mw_evecs_mode17"
MW_META_CSV   = "index_and_freq.csv"


def _mass_list(mdict, elems):
    return np.array([mdict[e] for e in elems], float)


def _sym(H):
    return 0.5 * (H + H.T)


def _norm(v):
    return math.sqrt(float(np.dot(v, v)))


def _nearest_image_vec(dr, box_lengths):
    if box_lengths is None:
        return dr
    out = dr.copy()
    for k, L in enumerate(box_lengths):
        if L and L > 0:
            out[k] -= L * round(out[k] / L)
    return out


def stream_hessian(filepath):
    """
    Streaming parser for a Hessian dump.
    Does not rely on an 'ndofs' header line.
    Yields (timestep, ndofs, H) where H is (ndofs x ndofs).
    """
    def _try_floats(s):
        try:
            return [float(x) for x in s.strip().split()]
        except ValueError:
            return None

    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            if "timestep" in line.lower():
                ts = None
                m = re.search(r'(-?\d+)', line)
                if m:
                    ts = int(m.group(1))
                else:
                    peek = f.readline()
                    if not peek:
                        break
                    m2 = re.search(r'(-?\d+)', peek)
                    if m2:
                        ts = int(m2.group(1))
                    else:
                        line = peek
                        continue

                first_vals = None
                while True:
                    row = f.readline()
                    if not row:
                        return
                    vals = _try_floats(row)
                    if vals is None:
                        continue
                    first_vals = vals
                    break

                ndofs = len(first_vals)
                rows = [first_vals]

                for _ in range(ndofs - 1):
                    row = f.readline()
                    if not row:
                        raise ValueError(
                            f"Unexpected EOF while reading Hessian block at timestep {ts}."
                        )
                    vals = _try_floats(row)
                    if vals is None or len(vals) != ndofs:
                        raise ValueError(
                            f"Inconsistent Hessian row at timestep {ts}: "
                            f"expected {ndofs} floats, got {len(vals) if vals else 'non-numeric'}."
                        )
                    rows.append(vals)

                H = np.array(rows, dtype=float)
                yield ts, ndofs, H

            line = f.readline()


class TrajStreamer:
    """
    Stream a large .lammpstrj file and provide nearest-frame lookup
    without indexing the whole file.

    Assumptions:
      - First n_solute ATOMS lines are the solute, in the desired order.
      - Orthorhombic box bounds with x/y/z or xs/ys/zs.
    """
    def __init__(self, path, n_solute=12):
        self.f = open(path, 'r')
        self.n_solute = n_solute
        self.prev = None
        self.curr = None
        self.eof = False
        self._read_next_frame()

    def close(self):
        try:
            self.f.close()
        except:
            pass

    def _read_next_frame(self):
        f = self.f
        line = f.readline()
        while line and not line.startswith("ITEM: TIMESTEP"):
            line = f.readline()
        if not line:
            self.eof = True
            return False

        ts = int(f.readline().strip())
        assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
        nall = int(f.readline().strip())

        box_hdr = f.readline().strip()
        assert box_hdr.startswith("ITEM: BOX BOUNDS")
        xlo, xhi = map(float, f.readline().split()[:2])
        ylo, yhi = map(float, f.readline().split()[:2])
        zlo, zhi = map(float, f.readline().split()[:2])
        Lx, Ly, Lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)

        atoms_hdr = f.readline().strip()
        assert atoms_hdr.startswith("ITEM: ATOMS")
        cols = atoms_hdr.split()[2:]
        has_cart = all(c in cols for c in ("x", "y", "z"))
        has_scaled = all(c in cols for c in ("xs", "ys", "zs"))
        if not (has_cart or has_scaled):
            raise RuntimeError("Need x y z OR xs ys zs columns in trajectory.")

        if has_cart:
            ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
            scaler = None
        else:
            ix, iy, iz = cols.index("xs"), cols.index("ys"), cols.index("zs")
            scaler = (xlo, ylo, zlo, Lx, Ly, Lz)

        coords = np.empty((self.n_solute, 3), float)
        for i in range(self.n_solute):
            parts = f.readline().split()
            x = float(parts[ix])
            y = float(parts[iy])
            z = float(parts[iz])
            if scaler:
                x = scaler[0] + x * scaler[3]
                y = scaler[1] + y * scaler[4]
                z = scaler[2] + z * scaler[5]
            coords[i, 0] = x
            coords[i, 1] = y
            coords[i, 2] = z

        for _ in range(nall - self.n_solute):
            f.readline()

        self.prev = self.curr
        self.curr = (ts, coords, (Lx, Ly, Lz))
        return True

    def nearest_for(self, target_ts):
        """
        Advance until curr.ts >= target_ts (or EOF).
        Return the closer of prev and curr.
        """
        if self.curr is None and not self.eof:
            self._read_next_frame()
        while (not self.eof) and (self.curr[0] < target_ts):
            self._read_next_frame()
        if self.prev is None:
            return self.curr
        prev_dt = abs(self.prev[0] - target_ts)
        curr_dt = abs(self.curr[0] - target_ts)
        return self.prev if prev_dt <= curr_dt else self.curr


def mass_weight_hessian(H, masses):
    m = np.array(masses, float)
    sqrtm = np.repeat(np.sqrt(m), 3)
    inv = 1.0 / sqrtm
    return H * np.outer(inv, inv)


def compute_modes(H, masses):
    D = mass_weight_hessian(_sym(H), masses)
    eigvals, Vmw = eigh(D)
    freqs = np.sign(eigvals) * FREQ_CONV * np.sqrt(np.abs(eigvals))
    return freqs, Vmw, eigvals


def mw_to_real(Vmw, masses):
    sqrtm = np.repeat(np.sqrt(np.array(masses, float)), 3)
    Vreal = Vmw / sqrtm[:, None]
    return Vreal.reshape((NUM_SOLUTE_ATOMS, 3, -1))


def local_CO_phva(H, masses_phys, idxC, idxO):
    dof = []
    for a in (idxC, idxO):
        base = 3 * a
        dof.extend([base, base + 1, base + 2])
    Hsub = H[np.ix_(dof, dof)]
    Dsub = mass_weight_hessian(_sym(Hsub), [masses_phys[idxC], masses_phys[idxO]])
    lam, _ = eigh(Dsub)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return FREQ_CONV * np.sqrt(lam.max())


def mode_character_score(disp_real, masses_energy, coords, box_lengths,
                         idxC, idxO, nh_indices, perp_penalty=PERP_PENALTY):
    N, _, K = disp_real.shape
    m = np.array(masses_energy, float)

    rC = coords[idxC]
    rO = coords[idxO]
    dCO = _nearest_image_vec(rO - rC, box_lengths)
    uCO = dCO / (_norm(dCO) + 1e-15)

    dC = disp_real[idxC, :, :]
    dO = disp_real[idxO, :, :]
    drel = dO - dC

    par = np.abs(np.einsum('i,ik->k', uCO, drel))
    par_vec = np.outer(uCO, np.einsum('i,ik->k', uCO, drel))
    perp_ratio = (np.linalg.norm(drel - par_vec, axis=0) /
                  (np.linalg.norm(drel, axis=0) + 1e-15))

    norms2 = np.sum(disp_real ** 2, axis=1)
    E = m[:, None] * norms2
    Etot = np.sum(E, axis=0) + 1e-15
    frac_CO = (E[idxC, :] + E[idxO, :]) / Etot
    frac_NH = (np.sum(E[nh_indices, :], axis=0) / Etot) if nh_indices else np.zeros_like(frac_CO)

    score = par * frac_CO - NH_PENALTY * frac_NH - perp_penalty * perp_ratio
    return score, frac_CO, frac_NH, par, perp_ratio


def pick_amideI_index(freqs_id, Vmw_id, masses_for_disp, masses_for_energy,
                      coords, box_lengths, prev_u_phys,
                      idxC, idxO, nh_indices, base_win, dyn_center, dyn_half, overlap_blend):
    disp = mw_to_real(Vmw_id, masses_for_disp)
    score, fCO, fNH, par, perp = mode_character_score(
        disp, masses_for_energy, coords, box_lengths, idxC, idxO, nh_indices
    )

    lo, hi = base_win
    if dyn_center is not None and math.isfinite(dyn_center):
        lo = max(lo, dyn_center - dyn_half)
        hi = min(hi, dyn_center + dyn_half)
    cand = np.where((freqs_id >= lo) & (freqs_id <= hi))[0]
    if cand.size == 0:
        cand = np.where(freqs_id > 10.0)[0]
        if cand.size == 0:
            cand = np.arange(len(freqs_id))

    s = score
    s_min, s_max = np.min(s[cand]), np.max(s[cand])
    s_norm = np.zeros_like(s)
    if s_max > s_min:
        s_norm = (s - s_min) / (s_max - s_min)

    final_metric = s_norm
    if prev_u_phys is not None:
        overlaps = np.abs(Vmw_id.T @ prev_u_phys)
        o = overlaps
        o_min, o_max = np.min(o[cand]), np.max(o[cand])
        o_norm = np.zeros_like(o)
        if o_max > o_min:
            o_norm = (o - o_min) / (o_max - o_min)
        final_metric = overlap_blend * o_norm + (1.0 - overlap_blend) * s_norm

    best = int(cand[np.argmax(final_metric[cand])])
    details = dict(
        frac_CO=fCO[best],
        frac_NH=fNH[best],
        par_amp=par[best],
        perp_ratio=perp[best],
        metric=final_metric[best],
        window=(lo, hi),
    )
    return best, details


def main():
    t0 = time.time()
    print(f"Streaming trajectory '{TRAJ_FILE}' and Hessian '{HESSIAN_FILE}'")

    masses_phys_atoms = _mass_list(PHYS_MASS, SOLUTE_ELEMENTS)
    masses_id_atoms = _mass_list(ID_MASS, SOLUTE_ELEMENTS)
    nh_indices = [i for i, e in enumerate(SOLUTE_ELEMENTS) if e in ("N", "H", "D")]

    ts_stream = TrajStreamer(TRAJ_FILE, n_solute=NUM_SOLUTE_ATOMS)

    if SAVE_MW_EVECS:
        outdir = Path(MW_EVECS_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        meta_path = outdir / MW_META_CSV
        if not meta_path.exists():
            with open(meta_path, "w") as m:
                m.write("timestep_hess,mode_index_phys,freq_cm-1\n")

    all_pinned = {}
    tracked = []
    prev_u_phys = None

    try:
        for ts_h, ndofs, H in stream_hessian(HESSIAN_FILE):
            if ndofs != 3 * NUM_SOLUTE_ATOMS:
                print(f"Warning: Hessian ts={ts_h}: ndofs={ndofs}, expected {3 * NUM_SOLUTE_ATOMS}")

            frame = ts_stream.nearest_for(ts_h)
            if frame is None:
                print(f"No trajectory frame available for ts={ts_h}; skipping.")
                continue
            ts_traj, coords, box = frame
            if abs(ts_traj - ts_h) > WARN_MISMATCH_STEPS:
                print(f"Warning: ts={ts_h}: nearest traj ts={ts_traj} (Δ={ts_traj - ts_h})")

            freqs_id, Vmw_id, _ = compute_modes(H, masses_id_atoms)

            local_ref = local_CO_phva(H, masses_phys_atoms, IDX_C, IDX_O)

            best_id, det = pick_amideI_index(
                freqs_id, Vmw_id, masses_id_atoms, masses_phys_atoms,
                coords, box, prev_u_phys,
                IDX_C, IDX_O, nh_indices,
                BASE_WINDOW, local_ref, DYN_HALF_WIDTH, OVERLAP_BLEND
            )

            freqs_phys, Vmw_phys, _ = compute_modes(H, masses_phys_atoms)

            u_id = Vmw_id[:, best_id]
            k_phys = int(np.argmax(np.abs(Vmw_phys.T @ u_id)))
            tracked_freq = float(freqs_phys[k_phys])

            if SAVE_MW_EVECS:
                evec_mw = Vmw_phys[:, k_phys].astype(np.float64)
                np.save(Path(MW_EVECS_DIR) / f"evec_mw_ts_{ts_h}.npy", evec_mw)
                with open(Path(MW_EVECS_DIR) / MW_META_CSV, "a") as m:
                    m.write(f"{ts_h},{k_phys},{tracked_freq:.8f}\n")

            freqs_list = freqs_phys.tolist()
            amide = freqs_list.pop(k_phys)
            rest_sorted = sorted(freqs_list)
            insert_at = min(max(AMIDE_I_TARGET_INDEX, 0), len(rest_sorted))
            rest_sorted.insert(insert_at, amide)
            all_pinned[ts_h] = np.array(rest_sorted, float)

            prev_u_phys = Vmw_phys[:, k_phys]

            naive = np.sort(freqs_phys)
            if insert_at < len(naive) and not np.isclose(naive[insert_at], tracked_freq, atol=1e-6):
                print(
                    f"Swap at ts={ts_h}: true Amide I {tracked_freq:.2f} cm-1 "
                    f"not at Mode {insert_at + 1}; pinned."
                )

            tracked.append((
                ts_h, insert_at + 1, tracked_freq,
                det["frac_CO"], det["frac_NH"], det["par_amp"], det["perp_ratio"],
                det["metric"], (local_ref if math.isfinite(local_ref) else float('nan')),
                det["window"][0], det["window"][1], ts_traj
            ))
    finally:
        ts_stream.close()

    print(f"\nWriting pinned frequencies to {PINNED_ALL_FILE}")
    with open(PINNED_ALL_FILE, "w") as f:
        f.write(
            f"# Timestep, Mode_Index, Frequency_cm-1 "
            f"(Amide I pinned to Mode {AMIDE_I_TARGET_INDEX + 1})\n"
        )
        for ts in sorted(all_pinned.keys()):
            arr = all_pinned[ts]
            for i, w in enumerate(arr, start=1):
                f.write(f"{ts}, {i}, {w:.6f}\n")

    print(f"Writing tracked Amide I to {TRACKED_ONLY_FILE}")
    with open(TRACKED_ONLY_FILE, "w") as g:
        g.write(
            "timestep,mode_index,amideI_cm1,frac_CO,frac_NH,par_amp,perp_ratio,"
            "metric,local_CO_ref_cm1,win_lo,win_hi,nearest_traj_ts\n"
        )
        for row in sorted(tracked, key=lambda r: r[0]):
            g.write(",".join([
                str(row[0]), str(row[1]),
                f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}",
                f"{row[5]:.6f}", f"{row[6]:.6f}", f"{row[7]:.6f}",
                (f"{row[8]:.6f}" if math.isfinite(row[8]) else "nan"),
                f"{row[9]:.2f}", f"{row[10]:.2f}", str(row[11])
            ]) + "\n")

    print(f"Done in {time.time() - t0:.2f} s.")
    if SAVE_MW_EVECS:
        print(f"Mass-weighted eigenvectors saved per timestep in '{MW_EVECS_DIR}/evec_mw_ts_<ts>.npy'")
        print(f"Metadata appended to '{MW_EVECS_DIR}/{MW_META_CSV}'")


if __name__ == "__main__":
    main()

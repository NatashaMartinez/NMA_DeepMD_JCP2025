import os
import numpy as np
from scipy.linalg import eigh  # For eigenvalue decomposition
from scipy.spatial.transform import Rotation as R

# File paths
processed_dir = "./processed_trajectory"
output_dir = "./dipole"
os.makedirs(output_dir, exist_ok=True)

# Dipole model file
dipole_model_file = "dipolemodel.pb"

# Load processed trajectory data in memory-mapped mode
coords_file = os.path.join(processed_dir, "coord.npy")
boxes_file = os.path.join(processed_dir, "box.npy")
atom_types_file = os.path.join(processed_dir, "type.raw")

# Ensure files exist
assert os.path.exists(coords_file), f"Missing file: {coords_file}"
assert os.path.exists(boxes_file), f"Missing file: {boxes_file}"
assert os.path.exists(atom_types_file), f"Missing file: {atom_types_file}"

# Memory-map large arrays
coords = np.load(coords_file, mmap_mode='r')  # shape: (n_frames, n_atoms*3)
boxes_raw = np.load(boxes_file, mmap_mode='r')  # shape: (n_frames, 6)
atom_types = np.loadtxt(atom_types_file, dtype=int)  # relatively small, load normally

# Debug: shapes
print(f"Coords shape: {coords.shape}")
print(f"Boxes raw shape: {boxes_raw.shape}")
print(f"Atom types shape: {atom_types.shape}")

num_frames = coords.shape[0]
n_atoms = 684
assert coords.shape[1] == n_atoms * 3, "Unexpected coords shape"
assert atom_types.shape[0] == n_atoms, "Unexpected atom_types length"

# Vectorized conversion of boxes_raw to 9D boxes:
# We know that box_matrix:
# [ [lx,   0,   0],
#   [  0,  ly,   0],
#   [  0,   0,  lz] ]
# So the flattened box for each frame is [lx, 0, 0, 0, ly, 0, 0, 0, lz].
boxes = np.zeros((num_frames, 9), dtype=boxes_raw.dtype)
boxes[:, 0] = boxes_raw[:, 0]  # lx
boxes[:, 4] = boxes_raw[:, 1]  # ly
boxes[:, 8] = boxes_raw[:, 2]  # lz

print(f"Converted boxes shape: {boxes.shape}")
print(f"Number of frames: {num_frames}")

# Load the dipole model
from deepmd.infer.deep_dipole import DeepDipole
dp = DeepDipole(dipole_model_file)

batch_size = 128
num_batches = (num_frames + batch_size - 1) // batch_size

dipoles_lab_frame = np.zeros((num_frames, 3), dtype=np.float64)

# Process dipoles in batches
for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min(start_idx + batch_size, num_frames)
    print(f"Processing batch {b+1}/{num_batches}: frames {start_idx} to {end_idx}")

    batch_coords = coords[start_idx:end_idx].reshape(-1, n_atoms, 3)
    batch_boxes = boxes[start_idx:end_idx]

    batch_dipoles = dp.eval(batch_coords.reshape(-1, n_atoms*3), batch_boxes, atom_types)

    if batch_dipoles.ndim == 3:
        batch_dipoles = np.sum(batch_dipoles, axis=1)  # sum over atoms

    dipoles_lab_frame[start_idx:end_idx] = batch_dipoles

# Save dipoles in the lab frame
dipole_lab_frame_file = os.path.join(output_dir, "dipoles_lab_frame_0.npy")
np.save(dipole_lab_frame_file, dipoles_lab_frame)
print("Saved dipoles in lab frame.")

# HERE, MOL FRAME DIPOLES WERE NOT USED SO PART BELOW MIGHT BE WRONG
# Transform dipoles to molecular frame
# First determine the reference frame from the first frame.
coords_first = coords[0].reshape(n_atoms, 3)
com = np.mean(coords_first, axis=0)
ref_displacements = coords_first - com

inertia_tensor = np.zeros((3, 3))
for r in ref_displacements:
    inertia_tensor += np.dot(r, r) * np.eye(3) - np.outer(r, r)

_, ref_axes = eigh(inertia_tensor)

# Pre-allocate molecular frame dipoles
dipoles_molecular_frame = np.zeros((num_frames, 3), dtype=np.float64)

for i in range(num_frames):
    if i % 10000 == 0 and i > 0:
        print(f"Transforming frame {i}/{num_frames}...")

    coords_frame = coords[i].reshape(n_atoms, 3)
    coords_displacements = coords_frame - np.mean(coords_frame, axis=0)

    H = ref_displacements.T @ coords_displacements
    U, S, Vt = np.linalg.svd(H)
    rot_matrix = U @ Vt

    dipole_lab = dipoles_lab_frame[i]
    rotated_dipole = np.dot(rot_matrix, dipole_lab)

    dipole_mol = np.dot(ref_axes.T, rotated_dipole)
    dipoles_molecular_frame[i] = dipole_mol

dipole_mol_frame_file = os.path.join(output_dir, "dipoles_molecular_frame_0.npy")
np.save(dipole_mol_frame_file, dipoles_molecular_frame)
print("Dipoles transformed to molecular frame and saved.")


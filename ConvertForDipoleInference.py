import numpy as np
import os

trajectory_file = "NewProd_5v4.lammpstrj"
output_dir = "./processed_trajectory"
os.makedirs(output_dir, exist_ok=True)

num_atoms = 684  # Adjust based on your system
batch_size = 256  # Number of frames per batch for mini-batching

def parse_lammps_trajectory(file_path, num_atoms, batch_size):
    """Parse LAMMPS trajectory file and convert it into analyzable format without loading entire file at once."""
    coords = []   # To store coordinates for all frames
    boxes = []    # To store box dimensions for all frames
    atom_types = None
    current_coords = []
    
    frame_count = 0
    reading_box = False
    reading_atoms = False
    atoms_to_read = 0
    box_lines_to_read = 0
    box_data = []
    
    with open(file_path, "r") as file:
        for line in file:
            line_strip = line.strip()
            
            if line_strip.startswith("ITEM:"):
                if "TIMESTEP" in line_strip:
                    if len(current_coords) == num_atoms:
                        coords.append(np.array(current_coords).flatten())
                        current_coords = []
                    
                    frame_count += 1
                    if frame_count % batch_size == 0:
                        print(f"Processed {frame_count} frames...")
                    
                    reading_box = False
                    reading_atoms = False
                    atoms_to_read = 0
                    box_lines_to_read = 0
                    box_data = []

                elif "BOX BOUNDS" in line_strip:
                    reading_box = True
                    box_lines_to_read = 3
                    reading_atoms = False
                    box_data = []

                elif "ATOMS" in line_strip:
                    reading_atoms = True
                    atoms_to_read = num_atoms
                    reading_box = False

            else:
                if reading_box and box_lines_to_read > 0:
                    parts = line_strip.split()
                    if len(parts) == 2 or len(parts) == 3: 
                        box_data.append([float(parts[0]), float(parts[1])])
                    box_lines_to_read -= 1
                    
                    if box_lines_to_read == 0:
                        lx = box_data[0][1] - box_data[0][0]
                        ly = box_data[1][1] - box_data[1][0]
                        lz = box_data[2][1] - box_data[2][0]
                        boxes.append([lx, ly, lz, 0.0, 0.0, 0.0])
                        reading_box = False

                elif reading_atoms and atoms_to_read > 0:
                    parts = line_strip.split()
                    if len(parts) >= 5:
                        atom_id = int(parts[0])
                        a_type = int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        current_coords.append([x, y, z])

                        if atom_types is None:
                            if atom_id == 1:
                                atom_types = []
                            atom_types.append(a_type)
                    atoms_to_read -= 1
                    if atoms_to_read == 0:
                        reading_atoms = False
    
    if len(current_coords) == num_atoms:
        coords.append(np.array(current_coords).flatten())

    print(f"Total frames processed: {frame_count}")
    return np.array(coords), np.array(boxes), np.array(atom_types)


# Convert LAMMPS trajectory to required format
coords, boxes, atom_types = parse_lammps_trajectory(trajectory_file, num_atoms, batch_size)

# Save processed data
print("Saving processed data...")
np.save(os.path.join(output_dir, "coord.npy"), coords)
np.save(os.path.join(output_dir, "box.npy"), boxes)


print(f"Processed trajectory saved in: {output_dir}")
print(f"Coordinates shape: {coords.shape}")
print(f"Box shape: {boxes.shape}")
print(f"Atom types shape: {atom_types.shape}")

print("Example: Mini-batching for large trajectories...")
num_frames = coords.shape[0]
for start_idx in range(0, num_frames, batch_size):
    end_idx = min(start_idx + batch_size, num_frames)
    batch_coords = coords[start_idx:end_idx]
    batch_boxes = boxes[start_idx:end_idx]

    print(f"Processing batch {start_idx // batch_size + 1}: Frames {start_idx} to {end_idx}")


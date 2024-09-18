from Bio import PDB
import numpy as np
import gudhi
import matplotlib.pyplot as plt

# Define functions
def read_pdb(file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    return structure

def extract_coordinates(structure):
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
    return np.array(coords)

def compute_persistent_homology(coords, max_edge_length=5.0):
    # Create a Rips complex from the coordinates
    rips_complex = gudhi.RipsComplex(points=coords, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Compute the persistence diagram
    persistence = simplex_tree.persistence()
    return persistence

def plot_persistence_diagram(persistence):
    # Separate birth and death values
    birth = [p[1][0] for p in persistence]
    death = [p[1][1] for p in persistence]

    plt.figure(figsize=(10, 6))
    plt.scatter(birth, death, marker='o')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence Diagram')
    plt.grid(True)
    plt.show()

# Main execution
pdb_file = '1bta.pdb'
structure = read_pdb(pdb_file)
coords = extract_coordinates(structure)

# Compute persistent homology
persistence = compute_persistent_homology(coords)

# Print and plot persistence diagram
print(f'Persistence diagram: {persistence}')
plot_persistence_diagram(persistence)

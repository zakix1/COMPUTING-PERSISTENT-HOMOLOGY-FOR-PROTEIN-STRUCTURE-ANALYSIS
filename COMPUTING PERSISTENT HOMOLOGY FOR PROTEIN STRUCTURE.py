import os
import numpy as np
from Bio import PDB
import gudhi
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def read_pdb(file_path):
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        if structure:
            print(f"Successfully read structure from {file_path}")
        return structure
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_multiple_pdbs(directory):
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    if not pdb_files:
        print("No PDB files found in the directory.")
    structures = {}
    for pdb_file in pdb_files:
        file_path = os.path.join(directory, pdb_file)
        structure = read_pdb(file_path)
        if structure:
            structures[pdb_file] = structure
        else:
            print(f"Failed to read structure from {file_path}.")
    return structures

def extract_coordinates(structure):
    coords = [atom.get_coord() for atom in structure.get_atoms()]
    if not coords:
        print("Warning: No coordinates found.")
    else:
        print(f"Extracted {len(coords)} coordinates.")
    return np.array(coords)

def reduce_dimensions(coords, n_components=3):
    if coords.shape[0] > 1:
        pca = PCA(n_components=n_components)
        coords_reduced = pca.fit_transform(coords)
        print(f"Reduced dimensions to {n_components}.")
    else:
        coords_reduced = coords
        print(f"Skipped PCA, coordinates are too few.")
    return coords_reduced

def subset_coordinates(coords, max_points=1000):
    if coords.shape[0] > max_points:
        indices = np.random.choice(coords.shape[0], max_points, replace=False)
        coords_subset = coords[indices]
        print(f"Subsampled coordinates to {max_points} points.")
    else:
        coords_subset = coords
        print(f"Used all {coords.shape[0]} coordinates.")
    return coords_subset

def compute_persistent_homology():
    try:
        coords_reduced = reduce_dimensions(coords, n_components=3)
        coords_sub = subset_coordinates(coords_reduced, max_points=500)

        if coords_sub.size == 0:
            print("Warning: Coordinates subset is empty after reduction.")
            return []

        print(f"Computing persistent homology for {len(coords_sub)} coordinates.")
        rips_complex = gudhi.RipsComplex(points=coords_sub)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

        persistence = simplex_tree.persistence()

        features = []
        for dim, persistence_intervals in persistence:
            for interval in persistence_intervals:
                if isinstance(interval, tuple) and len(interval) == 2:
                    birth, death = interval
                    if isinstance(birth, (int, float)) and isinstance(death, (int, float)):
                        features.append(death - birth)
                else:
                    print(f"Unexpected interval format: {interval}")

        print(f"Computed {len(features)} persistence features.")
        return features
    except MemoryError:
        print("Memory error during homology computation.")
        return []
    except Exception as e:
        print(f"Error during homology computation: {e}")
        return []

def process_pdb_files(directory):
    structures = read_multiple_pdbs(directory)
    homology_data = {}
    if not structures:
        print("No PDB files found or failed to read files.")
        return homology_data
    for pdb_file, structure in structures.items():
        print(f"Processing {pdb_file}...")
        coords = extract_coordinates(structure)
        if coords.size == 0:
            print(f"No coordinates extracted from {pdb_file}.")
            continue
        features = compute_persistent_homology(coords)
        if features:
            homology_data[pdb_file] = features
            print(f"Features for {pdb_file}: {features}")
        else:
            print(f"No features computed for {pdb_file}.")
    return homology_data

def prepare_data_for_ml(homology_data):
    features = list(homology_data.values())
    if not features:
        print("No homology data to prepare for ML.")
        return np.array([])

    max_length



# Step 2: Prepare data for ML


        # Example labels
    labels = np.array([0, 1] * (len(homology_features) // 2))  # Adjust based on your actual labels
    print(f"Labels: {labels}")

        # Step 3: Dimensionality Reduction and Model Training
    try:
        n_samples, n_features = homology_features.shape
        if n_samples > 1 and n_features > 1:
            pca = PCA(n_components=min(50, n_features))
            reduced_features = pca.fit_transform(homology_features)
            print(f"Reduced features shape: {reduced_features.shape}")
        else:
            reduced_features = homology_features
            print(f"Skipped PCA, features shape: {reduced_features.shape}")

            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(reduced_features)
            print(f"Normalized features shape: {normalized_features.shape}")

            X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure()
            plt.title("Feature Importances")
            plt.bar(range(reduced_features.shape[1]), importances[indices], align="center")
            plt.xticks(range(reduced_features.shape[1]), indices)
            plt.xlim([-1, reduced_features.shape[1]])
            plt.show()
    except Exception as e:
        print(f"Error during ML processing: {e}")

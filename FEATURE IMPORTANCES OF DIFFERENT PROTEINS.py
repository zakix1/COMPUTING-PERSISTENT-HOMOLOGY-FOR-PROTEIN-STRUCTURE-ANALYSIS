import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Example function to prepare data for ML
def prepare_data_for_ml(homology_data):
    max_length = max(len(v) for v in homology_data.values())
    homology_features = []
    for v in homology_data.values():
        if len(v) < max_length:
            padded_v = v + [0] * (max_length - len(v))
        else:
            padded_v = v[:max_length]
        homology_features.append(padded_v)
    return np.array(homology_features)

# Example data
homology_data = {
    '1bta.pdb': np.random.rand(1000),
    '1NOT.pdb': np.random.rand(1000),
    '1GK7.pdb': np.random.rand(1000),
    '1NKD.pdb': np.random.rand(1000),
    '2hbs.pdb': np.random.rand(1000),
    '2JOX.pdb': np.random.rand(1000),
    '2OL9.pdb': np.random.rand(1000),
    '3MD4.pdb': np.random.rand(1000),
    '3Q2X.pdb': np.random.rand(1000),
    '4AXY.pdb': np.random.rand(1000),
    '1DBP.pdb': np.random.rand(1000),
    '7UI2.pdb': np.random.rand(1000),
    '1HHP.pdb': np.random.rand(1000),
    'IHJEY.pdb': np.random.rand(1000),
    '1A3N.pdb': np.random.rand(1000),
    '1J4N.pdb': np.random.rand(1000),
    '3UON.pdb': np.random.rand(1000),
    '1DK8.pdb': np.random.rand(1000),
    '7UJ2.pdb': np.random.rand(1000),
    '1DF4.pdb': np.random.rand(1000),
    '1bx7.pdb': np.random.rand(1000),
    '1COS.pdb': np.random.rand(1000),
    '1DGV.pdb': np.random.rand(1000),
    '1a3n.pdb': np.random.rand(1000),
    '7UG5.pdb': np.random.rand(1000),
    # Add more data as needed'
}

labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 ])  # Example labels; ensure this matches the number of proteins

# Prepare data
homology_features = prepare_data_for_ml(homology_data)
print(f"Shape of homology_features: {homology_features.shape}")

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(homology_features)

# Determine the number of PCA components
n_samples, n_features = homology_features.shape
n_components = min(20, n_samples, n_features)  # Set appropriate number of components
print(f"Number of PCA components: {n_components}")

pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(normalized_features)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

# Train Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(reduced_features.shape[1]), importances[indices], align="center")
plt.xticks(range(reduced_features.shape[1]), indices)
plt.xlim([-1, reduced_features.shape[1]])
plt.show()

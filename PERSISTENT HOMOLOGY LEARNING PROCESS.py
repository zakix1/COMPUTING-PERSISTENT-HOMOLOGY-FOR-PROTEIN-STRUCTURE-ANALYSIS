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


def generate_sample_data(n_samples=24, n_features=1000):
    return np.random.rand(n_samples, n_features)

homology_features = generate_sample_data()
labels = np.array([0, 1] * (len(homology_features) // 2))  # Ensure labels match sample count

# Standardize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(homology_features)

# PCA
pca = PCA(n_components=20)
reduced_features = pca.fit_transform(normalized_features)
print(f"Reduced features shape: {reduced_features.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Check sample predictions
for i in range(min(5, len(X_test))):
    print(f"Sample {i}: Features: {X_test[i]}, Label: {y_test[i]}, Predicted: {y_pred[i]}")
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


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()



from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(reduced_features.shape[1]), importances[indices], align="center")
plt.xticks(range(reduced_features.shape[1]), indices)
plt.xlim([-1, reduced_features.shape[1]])
plt.show()


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, reduced_features, labels, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores)}")



from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(reduced_features, labels)



clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_resampled, y_resampled)
print(f"Best Parameters: {grid_search.best_params_}")





from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(reduced_features, labels)




from sklearn.inspection import permutation_importance






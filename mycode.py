# =============================
# Iris Classification Analysis
# COMP 30043 Machine Intelligence Systems
# =============================

# --- Import Libraries ---
# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning components
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --- Load Dataset ---
# Load built-in Iris dataset from scikit-learn
iris = load_iris()

# Create DataFrame for easier manipulation and visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]  # Add species names

# --- Exploratory Data Analysis ---
# 1. Class Distribution Analysis - Check for balance/imbalance
plt.figure(figsize=(8,5))
species_count = df['Species'].value_counts()
plt.bar(species_count.index, species_count.values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Iris Species Distribution')
plt.ylabel('Count')
plt.savefig('species_distribution.png', dpi=300)
plt.close()  # Close plot to prevent display in notebook environments

# 2. Feature Correlation Analysis - Identify relationships between features
plt.figure(figsize=(10,8))
corr_matrix = df.corr(numeric_only=True)  # Calculate pairwise correlations
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# 3. Feature Distribution by Species - Understand feature ranges across classes
plt.figure(figsize=(12,8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2,2,i+1)
    sns.boxplot(x='Species', y=feature, data=df)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300)
plt.close()

# --- Data Preprocessing ---
# 1. Label Encoding: Convert categorical species names to numerical labels
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])
# Mapping: 0=setosa, 1=versicolor, 2=virginica

# 2. Feature-Target Separation
X = df.drop(['Species', 'Species_encoded'], axis=1)  # Features (sepal/petal measurements)
y = df['Species_encoded']  # Target (encoded species)

# 3. Stratified Train-Test Split: Maintain class distribution in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80-20 split
    stratify=y,         # Preserve class ratios
    random_state=42     # Reproducibility
)
# Result: Training (120 samples), Testing (30 samples)

# 4. Feature Scaling for Distance-Based Algorithms (k-NN only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler to training data
X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test data
# Note: Decision trees don't require scaling

# --- Decision Tree Implementation ---
# Initialize classifier with regularization parameters
dt = DecisionTreeClassifier(
    max_depth=3,        # Limit tree depth to prevent overfitting
    criterion='gini',   # Splitting criterion (Gini impurity)
    random_state=42     # Reproducibility
)

# Train model on unscaled data (trees are scale-invariant)
dt.fit(X_train, y_train)

# Generate predictions on test set
y_pred_dt = dt.predict(X_test)

# Evaluate performance
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')

# Visualize decision tree structure
plt.figure(figsize=(15,10))
plot_tree(dt, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True,    # Color-coding by class
          rounded=True)   # Aesthetic preference
plt.title('Decision Tree for Iris Classification')
plt.savefig('decision_tree.png', dpi=300)
plt.close()

# --- k-NN Implementation with k-Selection ---
# 1. Optimal k Selection using Cross-Validation
k_values = range(1, 15)  # Test k values from 1 to 14
cv_scores = []           # Store mean accuracy scores

# Evaluate each k value using 5-fold cross-validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Compute mean accuracy across validation folds
    scores = cross_val_score(
        knn, 
        X_train_scaled,   # Use scaled training data
        y_train, 
        cv=5,             # 5-fold cross-validation
        scoring='accuracy' # Evaluation metric
    )
    cv_scores.append(scores.mean())

# 2. k-Selection Visualization
plt.figure(figsize=(10,6))
plt.plot(k_values, cv_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('k Value (Number of Neighbors)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('k-NN Performance by Neighborhood Size')
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight optimal k
optimal_k = k_values[np.argmax(cv_scores)]
plt.plot(optimal_k, max(cv_scores), 'r*', markersize=15)
plt.annotate(f'Optimal k={optimal_k}\nAccuracy={max(cv_scores):.2%}',
             xy=(optimal_k, max(cv_scores)),
             xytext=(optimal_k+0.5, max(cv_scores)-0.02),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('k_selection.png', dpi=300)
plt.close()

# 3. Final k-NN Model Training
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)  # Train on scaled data
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate performance
knn_acc = accuracy_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

# --- Performance Comparison ---
# Decision Tree Metrics
print('='*50)
print('Decision Tree Performance:')
print('='*50)
print(f'Accuracy: {dt_acc:.4f}')
print(f'F1-Score: {dt_f1:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# k-NN Metrics
print('\n' + '='*50)
print(f'k-NN Performance (k={optimal_k}):')
print('='*50)
print(f'Accuracy: {knn_acc:.4f}')
print(f'F1-Score: {knn_f1:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# Confusion Matrix Visualization for k-NN
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('Predicted Species')
plt.ylabel('True Species')
plt.title(f'Confusion Matrix (k-NN, k={optimal_k})')
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
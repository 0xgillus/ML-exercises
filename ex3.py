import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# For faster experimentation, let's use a subset of the data for grid search
# We'll use the full dataset for final evaluation
print("Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# For grid search, use an even smaller subset to speed up computation
X_train_small, _, y_train_small, _ = train_test_split(
    X_train, y_train, train_size=0.1, random_state=42, stratify=y_train
)

print(f"Small training set shape: {X_train_small.shape}")
print(f"Full training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale the data (important for KNN)
print("Scaling features...")
scaler = StandardScaler()
X_train_small_scaled = scaler.fit_transform(X_train_small)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grid
print("Setting up grid search...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

# Create KNN classifier
knn = KNeighborsClassifier()

# Perform grid search with cross-validation
print("Performing grid search (this may take a while)...")
start_time = time.time()

grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=3,  # 3-fold cross-validation for speed
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Fit grid search on small training set
grid_search.fit(X_train_small_scaled, y_train_small)

grid_search_time = time.time() - start_time
print(f"Grid search completed in {grid_search_time:.2f} seconds")

# Get best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation score: {best_score:.4f}")

# Train final model with best parameters on full training set
print("Training final model on full training set...")
final_knn = KNeighborsClassifier(**best_params)

start_time = time.time()
final_knn.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"Final model training completed in {training_time:.2f} seconds")

# Make predictions on test set
print("Making predictions on test set...")
start_time = time.time()
y_pred = final_knn.predict(X_test_scaled)
prediction_time = time.time() - start_time

print(f"Predictions completed in {prediction_time:.2f} seconds")

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check if we achieved the target
if test_accuracy > 0.97:
    print("Achieved over 97% accuracy!")
else:
    print(f"Close! Need {(0.97 - test_accuracy)*100:.2f}% more to reach 97%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Show some example predictions
def show_example_predictions(X, y_true, y_pred, n_examples=10):
    """Show some example predictions"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Get some random indices
    indices = np.random.choice(len(X), n_examples, replace=False)
    
    for i, idx in enumerate(indices):
        # Reshape image back to 28x28
        image = X[idx].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}')
        axes[i].axis('off')
        
        # Color the title based on correctness
        if y_true[idx] == y_pred[idx]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.suptitle('Example Predictions (Green=Correct, Red=Incorrect)', y=1.02)
    plt.show()

print("\nShowing example predictions...")
show_example_predictions(X_test, y_test, y_pred)

# Performance summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(f"Best hyperparameters: {best_params}")
print(f"Cross-validation score: {best_score:.4f}")
print(f"Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Grid search time: {grid_search_time:.2f} seconds")
print(f"Training time: {training_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")

# Additional hyperparameter analysis
print("\nGrid Search Results Details:")
results_df = pd.DataFrame(grid_search.cv_results_)
for i in range(len(results_df)):
    params = results_df.iloc[i]['params']
    mean_score = results_df.iloc[i]['mean_test_score']
    std_score = results_df.iloc[i]['std_test_score']
    print(f"n_neighbors={params['n_neighbors']}, weights='{params['weights']}': "
          f"{mean_score:.4f} (+/- {std_score*2:.4f})")

print("\n" + "="*50)
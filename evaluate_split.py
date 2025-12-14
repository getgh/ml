"""
Evaluate classifier accuracy on split datasets with known test labels
"""
from classifier import load_data, load_labels, impute_missing_values, normalize_data, knn_predict

def accuracy_score(y_true, y_pred):
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

def evaluate_split_dataset(dataset_num, k=5):
    """Evaluate on the split dataset with known test labels"""
    train_data = f"TrainData{dataset_num}_split.txt"
    train_labels = f"TrainLabel{dataset_num}_split.txt"
    test_data = f"TestData{dataset_num}_split.txt"
    test_labels = f"TestLabel{dataset_num}_split.txt"
    
    X_train = load_data(train_data)
    y_train = load_labels(train_labels)
    X_test = load_data(test_data)
    y_test = load_labels(test_labels)
    
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # Test different k values
    results = []
    for k_val in [1, 3, 5, 7, 9]:
        if k_val >= len(X_train):
            break
        y_pred = knn_predict(X_train_norm, y_train, X_test_norm, k=k_val)
        acc = accuracy_score(y_test, y_pred)
        results.append((k_val, acc))
    
    return results, len(X_train), len(X_test)

def main():
    print("Evaluating classifier on split datasets with ground truth labels\n")
    print("=" * 70)
    
    for dataset_num in range(1, 5):
        try:
            results, n_train, n_test = evaluate_split_dataset(dataset_num)
            
            print(f"\nDataset {dataset_num}:")
            print(f"  Training samples: {n_train}")
            print(f"  Test samples: {n_test}")
            print(f"\n  Accuracy by k value:")
            
            for k, acc in results:
                print(f"    k={k}: {acc:.4f} ({acc*100:.2f}%)")
            
            # Find best k
            best_k, best_acc = max(results, key=lambda x: x[1])
            print(f"\n  Best: k={best_k} with {best_acc:.4f} ({best_acc*100:.2f}%) accuracy")
            print("-" * 70)
            
        except Exception as e:
            print(f"\nError evaluating dataset {dataset_num}: {e}")
            print("-" * 70)
    
    print("\n" + "=" * 70)
    print("\nTo improve accuracy further, try:")
    print("  1. Increase k for smoother decision boundaries")
    print("  2. Try different distance metrics (Manhattan)")
    print("  3. Use ensemble methods (multiple k values)")
    print("  4. Feature selection or PCA for high-dimensional data")

if __name__ == "__main__":
    main()

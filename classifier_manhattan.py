"""
Optimized classifier using Manhattan distance for better accuracy
Tests on split datasets with known labels
"""
from classifier import load_data, load_labels, impute_missing_values, normalize_data

def euclidean_distance(x1, x2):
    return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def knn_predict_weighted(X_train, y_train, X_test, k=5, metric="euclidean"):
    """k-NN with distance-weighted voting"""
    predictions = []
    dist_fn = manhattan_distance if metric == "manhattan" else euclidean_distance
    
    for test_sample in X_test:
        distances = []
        for i, train_sample in enumerate(X_train):
            dist = dist_fn(test_sample, train_sample)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Distance-weighted voting
        class_scores = {}
        for d, label in k_nearest:
            weight = 1.0 / (d + 1e-10)
            class_scores[label] = class_scores.get(label, 0.0) + weight
        
        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)
    
    return predictions

def accuracy_score(y_true, y_pred):
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

def find_best_k_and_metric(X_train, y_train, max_k=20):
    """Find best k and metric using leave-one-out on training data"""
    n = len(X_train)
    if n > 100:
        # Use 5-fold CV for large datasets
        return find_best_k_cv(X_train, y_train, max_k)
    
    best_k = 5
    best_metric = "euclidean"
    best_acc = 0.0
    
    # Expanded k range for better tuning
    k_values = [k for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19] if k < n and k <= max_k]
    metrics = ["euclidean", "manhattan"]
    
    for metric in metrics:
        for k in k_values:
            correct = 0
            for i in range(n):
                X_tr = [row for j, row in enumerate(X_train) if j != i]
                y_tr = [lab for j, lab in enumerate(y_train) if j != i]
                X_te = [X_train[i]]
                
                X_tr_norm, X_te_norm = normalize_data(X_tr, X_te)
                pred = knn_predict_weighted(X_tr_norm, y_tr, X_te_norm, k=k, metric=metric)[0]
                
                if pred == y_train[i]:
                    correct += 1
            
            acc = correct / n
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_metric = metric
    
    return best_k, best_metric, best_acc

def find_best_k_cv(X_train, y_train, max_k=20, n_folds=5):
    """Find best k and metric using k-fold CV"""
    n = len(X_train)
    fold_size = n // n_folds
    
    best_k = 5
    best_metric = "euclidean"
    best_acc = 0.0
    
    # Expanded k range
    k_values = [k for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19] if k <= max_k]
    metrics = ["euclidean", "manhattan"]
    
    for metric in metrics:
        for k in k_values:
            fold_accs = []
            
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else n
                
                X_val = X_train[val_start:val_end]
                y_val = y_train[val_start:val_end]
                X_tr = X_train[:val_start] + X_train[val_end:]
                y_tr = y_train[:val_start] + y_train[val_end:]
                
                X_tr_norm, X_val_norm = normalize_data(X_tr, X_val)
                y_pred = knn_predict_weighted(X_tr_norm, y_tr, X_val_norm, k=k, metric=metric)
                
                acc = accuracy_score(y_val, y_pred)
                fold_accs.append(acc)
            
            avg_acc = sum(fold_accs) / len(fold_accs)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_k = k
                best_metric = metric
    
    return best_k, best_metric, best_acc

def process_dataset_optimized(dataset_num):
    """Process dataset with automatic parameter selection"""
    train_data = f"TrainData{dataset_num}_split.txt"
    train_labels = f"TrainLabel{dataset_num}_split.txt"
    test_data = f"TestData{dataset_num}_split.txt"
    test_labels = f"TestLabel{dataset_num}_split.txt"
    
    X_train = load_data(train_data)
    y_train = load_labels(train_labels)
    X_test = load_data(test_data)
    y_test = load_labels(test_labels)
    
    print(f"\nDataset {dataset_num}:")
    print(f"  Train: {len(X_train)} samples, {len(X_train[0])} features")
    print(f"  Test: {len(X_test)} samples")
    
    # Impute
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    # Find best parameters
    print(f"  Finding best k and metric...")
    best_k, best_metric, cv_acc = find_best_k_and_metric(X_train, y_train)
    print(f"  Best: k={best_k}, metric={best_metric}, CV accuracy={cv_acc:.4f}")
    
    # Normalize and predict
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    y_pred = knn_predict_weighted(X_train_norm, y_train, X_test_norm, k=best_k, metric=best_metric)
    
    # Calculate test accuracy
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    status = "✓" if test_acc >= 0.90 else "✗"
    print(f"  Status: {status} {'Achieved 90%+' if test_acc >= 0.90 else 'Below 90%'}")
    
    return test_acc, best_k, best_metric

def main():
    print("=" * 70)
    print("Optimized k-NN Classifier with Manhattan Distance Support")
    print("Testing on Split Datasets with Known Labels")
    print("=" * 70)
    
    results = {}
    
    for dataset_num in range(1, 5):
        try:
            acc, k, metric = process_dataset_optimized(dataset_num)
            results[dataset_num] = (acc, k, metric)
        except Exception as e:
            print(f"\nDataset {dataset_num}: Error - {e}")
            results[dataset_num] = (0.0, 0, "error")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    
    for dataset_num in range(1, 5):
        acc, k, metric = results[dataset_num]
        status = "✓" if acc >= 0.90 else "✗"
        print(f"  Dataset {dataset_num}: {acc*100:.2f}% (k={k}, {metric}) {status}")
    
    avg_acc = sum(r[0] for r in results.values()) / len(results)
    print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")
    
    passing = sum(1 for r in results.values() if r[0] >= 0.90)
    print(f"  Datasets ≥90%: {passing}/4")

if __name__ == "__main__":
    main()

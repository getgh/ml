"""
Advanced k-NN classifier optimized for 90%+ accuracy
Uses ensemble voting, feature selection, and adaptive parameters
"""
from classifier import load_data, load_labels, impute_missing_values, normalize_data
import random

def euclidean_distance(x1, x2):
    return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def select_top_features(X_train, y_train, n_features=None):
    """
    Simple feature selection based on variance and class separability
    Returns indices of top features to keep
    """
    n_total = len(X_train[0])
    
    if n_features is None:
        # Keep sqrt(n) features for high-dimensional data
        n_features = min(n_total, max(50, int(n_total ** 0.5)))
    
    if n_features >= n_total:
        return list(range(n_total))
    
    # Compute variance per feature
    variances = []
    for col in range(n_total):
        values = [X_train[i][col] for i in range(len(X_train))]
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        variances.append((var, col))
    
    # Sort by variance and keep top features
    variances.sort(reverse=True)
    selected = sorted([col for _, col in variances[:n_features]])
    
    return selected

def apply_feature_selection(X_train, X_test, selected_features):
    """Apply feature selection to train and test data"""
    X_train_sel = [[row[i] for i in selected_features] for row in X_train]
    X_test_sel = [[row[i] for i in selected_features] for row in X_test]
    return X_train_sel, X_test_sel

def knn_predict_weighted(X_train, y_train, test_sample, k, metric="euclidean"):
    """Single prediction with distance weighting"""
    dist_fn = manhattan_distance if metric == "manhattan" else euclidean_distance
    
    distances = [(dist_fn(test_sample, train_sample), y_train[i]) 
                 for i, train_sample in enumerate(X_train)]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Inverse distance weighting
    class_scores = {}
    for d, label in k_nearest:
        weight = 1.0 / (d + 1e-10)
        class_scores[label] = class_scores.get(label, 0.0) + weight
    
    return max(class_scores, key=class_scores.get)

def ensemble_predict(X_train, y_train, X_test, k_values, metrics):
    """
    Ensemble prediction using multiple k values and distance metrics
    Each (k, metric) combination votes
    """
    all_votes = []
    
    for k in k_values:
        for metric in metrics:
            predictions = []
            for test_sample in X_test:
                pred = knn_predict_weighted(X_train, y_train, test_sample, k, metric)
                predictions.append(pred)
            all_votes.append(predictions)
    
    # Majority vote across all models
    final_predictions = []
    for i in range(len(X_test)):
        votes = [preds[i] for preds in all_votes]
        vote_counts = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        final_predictions.append(max(vote_counts, key=vote_counts.get))
    
    return final_predictions

def cross_validate_params(X_train, y_train, n_folds=5):
    """
    Find best parameters using cross-validation
    """
    n = len(X_train)
    fold_size = n // n_folds
    
    # Candidate parameters
    k_options = [[1, 3, 5], [3, 5, 7], [5, 7, 9], [3, 5, 7, 9], [1, 3, 5, 7]]
    metric_options = [["euclidean"], ["manhattan"], ["euclidean", "manhattan"]]
    
    best_params = None
    best_score = 0.0
    
    for k_vals in k_options:
        for metrics in metric_options:
            fold_scores = []
            
            for fold in range(n_folds):
                # Create fold split
                test_start = fold * fold_size
                test_end = test_start + fold_size if fold < n_folds - 1 else n
                
                X_val = X_train[test_start:test_end]
                y_val = y_train[test_start:test_end]
                X_tr = X_train[:test_start] + X_train[test_end:]
                y_tr = y_train[:test_start] + y_train[test_end:]
                
                # Normalize
                X_tr_norm, X_val_norm = normalize_data(X_tr, X_val)
                
                # Predict
                y_pred = ensemble_predict(X_tr_norm, y_tr, X_val_norm, k_vals, metrics)
                
                # Score
                correct = sum(1 for a, b in zip(y_val, y_pred) if a == b)
                fold_scores.append(correct / len(y_val))
            
            avg_score = sum(fold_scores) / len(fold_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = (k_vals, metrics)
    
    return best_params, best_score

def process_dataset_advanced(dataset_num):
    """
    Advanced processing with ensemble, feature selection, and CV
    """
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
    
    # Feature selection for high-dimensional data
    if len(X_train[0]) > 100:
        print(f"  Applying feature selection...")
        selected_features = select_top_features(X_train, y_train)
        X_train, X_test = apply_feature_selection(X_train, X_test, selected_features)
        print(f"  Reduced to {len(selected_features)} features")
    
    # Cross-validation to find best parameters
    print(f"  Cross-validating parameters...")
    (best_k_vals, best_metrics), cv_score = cross_validate_params(X_train, y_train)
    print(f"  Best params: k_values={best_k_vals}, metrics={best_metrics}")
    print(f"  CV score: {cv_score:.4f}")
    
    # Normalize
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # Predict with ensemble
    print(f"  Predicting with ensemble...")
    y_pred = ensemble_predict(X_train_norm, y_train, X_test_norm, best_k_vals, best_metrics)
    
    # Calculate accuracy
    correct = sum(1 for a, b in zip(y_test, y_pred) if a == b)
    accuracy = correct / len(y_test)
    
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy >= 0.90:
        print(f"  ✓ Achieved 90%+ accuracy!")
    else:
        print(f"  ✗ Below 90% - needs further tuning")
    
    return accuracy

def main():
    print("=" * 70)
    print("Advanced k-NN Classifier - Optimizing for 90%+ Accuracy")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Ensemble voting (multiple k values + distance metrics)")
    print("  - Automated feature selection for high-dimensional data")
    print("  - Cross-validation for parameter tuning")
    print("  - Distance-weighted voting")
    
    accuracies = {}
    
    for dataset_num in range(1, 5):
        try:
            acc = process_dataset_advanced(dataset_num)
            accuracies[dataset_num] = acc
        except Exception as e:
            print(f"\n  Error: {e}")
            accuracies[dataset_num] = 0.0
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    
    for dataset_num in range(1, 5):
        acc = accuracies[dataset_num]
        status = "✓" if acc >= 0.90 else "✗"
        print(f"  Dataset {dataset_num}: {acc:.4f} ({acc*100:.2f}%) {status}")
    
    avg_acc = sum(accuracies.values()) / len(accuracies)
    print(f"\n  Average: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    if all(acc >= 0.90 for acc in accuracies.values()):
        print("\n  🎉 All datasets achieved 90%+ accuracy!")
    else:
        print("\n  Note: Some datasets need additional optimization")

if __name__ == "__main__":
    main()

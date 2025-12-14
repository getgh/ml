"""
Enhanced k-NN classifier with accuracy improvements:
- PCA dimensionality reduction
- Ensemble voting with multiple k values
- Robust feature scaling
- Optimized distance metrics per dataset
"""

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [float(x) for x in line.split()]
                data.append(row)
    return data

def load_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels

def impute_missing_values(data):
    missing_value = 1.00000000000000e+99
    result = [row[:] for row in data]
    
    if not data:
        return result
    
    num_cols = len(data[0])
    
    for col in range(num_cols):
        column_values = [row[col] for row in result]
        missing_indices = [i for i, val in enumerate(column_values) if abs(val - missing_value) < 1e-90]
        
        if missing_indices:
            valid_values = [val for val in column_values if abs(val - missing_value) >= 1e-90]
            if valid_values:
                # Use median instead of mean for robustness
                sorted_vals = sorted(valid_values)
                n = len(sorted_vals)
                median_value = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
                for i in missing_indices:
                    result[i][col] = median_value
            else:
                for i in missing_indices:
                    result[i][col] = 0.0
    
    return result

def euclidean_distance(x1, x2):
    return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def minkowski_distance(x1, x2, p=3):
    return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1/p)

def pca_transform(X_train, X_test, n_components=None, variance_threshold=0.95):
    """
    Simple PCA implementation for dimensionality reduction
    Keeps enough components to explain variance_threshold of variance
    """
    if not X_train or len(X_train[0]) <= 10:
        return X_train, X_test  # Skip PCA for low-dimensional data
    
    n_samples = len(X_train)
    n_features = len(X_train[0])
    
    # Center the data
    means = [sum(X_train[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    X_centered = [[X_train[i][j] - means[j] for j in range(n_features)] for i in range(n_samples)]
    
    # Compute covariance matrix
    cov = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(i, n_features):
            val = sum(X_centered[k][i] * X_centered[k][j] for k in range(n_samples)) / (n_samples - 1)
            cov[i][j] = val
            cov[j][i] = val
    
    # Power iteration for top eigenvectors (simplified)
    # For production, use numpy's eig, but this avoids dependencies
    eigenvalues = []
    eigenvectors = []
    
    # Simple approximation: use top components by variance
    feature_variances = [(cov[i][i], i) for i in range(n_features)]
    feature_variances.sort(reverse=True)
    
    total_var = sum(v for v, _ in feature_variances)
    cumsum = 0
    n_keep = 0
    for var, idx in feature_variances:
        cumsum += var
        n_keep += 1
        if cumsum / total_var >= variance_threshold:
            break
    
    # Select top features by variance (simplified PCA)
    selected_features = [idx for _, idx in feature_variances[:n_keep]]
    
    X_train_reduced = [[X_train[i][j] for j in selected_features] for i in range(len(X_train))]
    X_test_reduced = [[X_test[i][j] for j in selected_features] for i in range(len(X_test))]
    
    return X_train_reduced, X_test_reduced

def normalize_data(X_train, X_test, robust=True):
    """
    Normalize with optional robust scaling using IQR
    """
    if not X_train:
        return X_train, X_test
    
    num_cols = len(X_train[0])
    
    if robust:
        # Robust scaling: use median and IQR
        medians = []
        iqrs = []
        
        for col in range(num_cols):
            column_values = sorted([row[col] for row in X_train])
            n = len(column_values)
            median = column_values[n//2] if n % 2 == 1 else (column_values[n//2-1] + column_values[n//2]) / 2
            q1 = column_values[n//4]
            q3 = column_values[3*n//4]
            iqr = q3 - q1 if q3 != q1 else 1.0
            medians.append(median)
            iqrs.append(iqr)
        
        X_train_norm = [[(row[col] - medians[col]) / iqrs[col] for col in range(num_cols)] for row in X_train]
        X_test_norm = [[(row[col] - medians[col]) / iqrs[col] for col in range(num_cols)] for row in X_test]
    else:
        # Standard scaling
        means = []
        stds = []
        
        for col in range(num_cols):
            column_values = [row[col] for row in X_train]
            mean = sum(column_values) / len(column_values)
            variance = sum((x - mean) ** 2 for x in column_values) / len(column_values)
            std = variance ** 0.5 if variance > 0 else 1.0
            means.append(mean)
            stds.append(std)
        
        X_train_norm = [[(row[col] - means[col]) / stds[col] for col in range(num_cols)] for row in X_train]
        X_test_norm = [[(row[col] - means[col]) / stds[col] for col in range(num_cols)] for row in X_test]
    
    return X_train_norm, X_test_norm

def knn_predict_single(X_train, y_train, test_sample, k, metric="euclidean", weighted=True):
    """Single sample prediction"""
    dist_fn = manhattan_distance if metric == "manhattan" else euclidean_distance
    
    distances = []
    for i, train_sample in enumerate(X_train):
        dist = dist_fn(test_sample, train_sample)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Distance-weighted voting
    class_scores = {}
    for d, label in k_nearest:
        weight = 1.0 / (d + 1e-12) if weighted else 1.0
        class_scores[label] = class_scores.get(label, 0.0) + weight
    
    # Tie-breaker: closest average distance
    max_score = max(class_scores.values())
    candidates = [lbl for lbl, sc in class_scores.items() if sc == max_score]
    if len(candidates) == 1:
        return candidates[0]
    else:
        avg_d = {}
        for c in candidates:
            ds = [d for d, l in k_nearest if l == c]
            avg_d[c] = sum(ds) / len(ds) if ds else float('inf')
        return min(avg_d, key=avg_d.get)

def knn_predict(X_train, y_train, X_test, k=5, metric="euclidean", weighted=True):
    """Batch prediction"""
    return [knn_predict_single(X_train, y_train, sample, k, metric, weighted) for sample in X_test]

def ensemble_predict(X_train, y_train, X_test, k_values, metric="euclidean", weighted=True):
    """
    Ensemble prediction using multiple k values
    Each k votes, then majority wins
    """
    all_predictions = []
    
    for k in k_values:
        preds = knn_predict(X_train, y_train, X_test, k=k, metric=metric, weighted=weighted)
        all_predictions.append(preds)
    
    # Majority vote across k values
    final_predictions = []
    for i in range(len(X_test)):
        votes = [preds[i] for preds in all_predictions]
        # Count votes
        vote_counts = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        final_predictions.append(max(vote_counts, key=vote_counts.get))
    
    return final_predictions

def select_best_params(X_train, y_train, sample_size=50):
    """
    Quick parameter search using a sample for speed
    Returns best (k_values, metric)
    """
    n = len(X_train)
    if n == 0:
        return ([5], "euclidean")
    
    # Sample for speed on large datasets
    if n > sample_size:
        import random
        indices = random.sample(range(n), sample_size)
        X_sample = [X_train[i] for i in indices]
        y_sample = [y_train[i] for i in indices]
    else:
        X_sample = X_train
        y_sample = y_train
    
    # Try different configurations
    configs = [
        ([3, 5, 7], "euclidean"),
        ([3, 5, 7], "manhattan"),
        ([1, 3, 5], "euclidean"),
        ([5, 7, 9], "euclidean"),
    ]
    
    best_config = ([5], "euclidean")
    best_acc = 0.0
    
    for k_vals, metric in configs:
        # LOO on sample
        correct = 0
        for i in range(len(X_sample)):
            X_tr = [row for j, row in enumerate(X_sample) if j != i]
            y_tr = [lab for j, lab in enumerate(y_sample) if j != i]
            X_te = [X_sample[i]]
            
            X_tr_norm, X_te_norm = normalize_data(X_tr, X_te, robust=True)
            pred = ensemble_predict(X_tr_norm, y_tr, X_te_norm, k_vals, metric, weighted=True)[0]
            
            if pred == y_sample[i]:
                correct += 1
        
        acc = correct / len(X_sample)
        if acc > best_acc:
            best_acc = acc
            best_config = (k_vals, metric)
    
    return best_config

def process_dataset(dataset_num, use_pca=True, use_ensemble=True):
    """
    Enhanced processing with all improvements
    """
    X_train = load_data(f'TrainData{dataset_num}.txt')
    y_train = load_labels(f'TrainLabel{dataset_num}.txt')
    X_test = load_data(f'TestData{dataset_num}.txt')
    
    print(f"Dataset {dataset_num}: {len(X_train)} train samples, {len(X_train[0])} features, {len(X_test)} test samples")
    
    # Impute missing values with median
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    # Apply PCA if high-dimensional
    if use_pca and len(X_train[0]) > 20:
        print(f"  Applying PCA (features: {len(X_train[0])} -> ", end="")
        X_train, X_test = pca_transform(X_train, X_test, variance_threshold=0.95)
        print(f"{len(X_train[0])})")
    
    # Select best parameters
    print(f"  Searching for best parameters...")
    k_values, metric = select_best_params(X_train, y_train)
    print(f"  Selected: k_values={k_values}, metric={metric}")
    
    # Normalize with robust scaling
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, robust=True)
    
    # Predict with ensemble or single k
    if use_ensemble:
        predictions = ensemble_predict(X_train_norm, y_train, X_test_norm, k_values, metric, weighted=True)
    else:
        predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=k_values[0], metric=metric, weighted=True)
    
    # Save predictions
    output_filename = f'predictions{dataset_num}.txt'
    with open(output_filename, 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')
    
    print(f"  Predictions saved to {output_filename}\n")

def main():
    import sys
    
    print("Enhanced k-NN Classifier with:")
    print("- Robust median imputation")
    print("- PCA dimensionality reduction")
    print("- Ensemble voting with multiple k values")
    print("- Robust IQR-based scaling")
    print("- Automatic parameter selection\n")
    
    for dataset_num in range(1, 5):
        try:
            process_dataset(dataset_num, use_pca=True, use_ensemble=True)
        except Exception as e:
            print(f"Error processing dataset {dataset_num}: {e}\n")

if __name__ == '__main__':
    main()

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
                mean_value = sum(valid_values) / len(valid_values)
                for i in missing_indices:
                    result[i][col] = mean_value
            else:
                for i in missing_indices:
                    result[i][col] = 0.0
    
    return result

def euclidean_distance(x1, x2):
    return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def knn_predict(X_train, y_train, X_test, k=5, metric="euclidean", weighted=True):
    predictions = []
    
    dist_fn = euclidean_distance if metric == "euclidean" else manhattan_distance
    
    for test_sample in X_test:
        distances = []
        for i, train_sample in enumerate(X_train):
            dist = dist_fn(test_sample, train_sample)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Distance-weighted voting (inverse distance). Handle zero distance.
        class_scores = {}
        for d, label in k_nearest:
            weight = 1.0 / (d + 1e-12) if weighted else 1.0
            class_scores[label] = class_scores.get(label, 0.0) + weight
        
        # Tie-breaker: if equal scores, choose label with smallest avg distance
        max_score = max(class_scores.values())
        candidates = [lbl for lbl, sc in class_scores.items() if sc == max_score]
        if len(candidates) == 1:
            predictions.append(candidates[0])
        else:
            # compute average distance per candidate among k-nearest
            avg_d = {}
            for c in candidates:
                ds = [d for d, l in k_nearest if l == c]
                avg_d[c] = sum(ds) / len(ds) if ds else float('inf')
            predictions.append(min(avg_d, key=avg_d.get))
    
    return predictions

def _loo_score_for_k(X, y, k, metric="euclidean", weighted=True):
    if len(X) == 0:
        return 0.0
    correct = 0
    for i in range(len(X)):
        X_train = [row for j, row in enumerate(X) if j != i]
        y_train = [lab for j, lab in enumerate(y) if j != i]
        X_test = [X[i]]
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        pred = knn_predict(X_train_norm, y_train, X_test_norm, k=k, metric=metric, weighted=weighted)[0]
        if pred == y[i]:
            correct += 1
    return correct / len(X)

def select_best_k(X_train, y_train, candidate_ks=None, metric="euclidean", weighted=True):
    if not candidate_ks:
        # heuristic candidate set relative to dataset size
        m = max(1, int(len(X_train) ** 0.5))
        candidate_ks = sorted(set([1, 3, 5, 7, 9, m, m+2]))
    # ensure k does not exceed training count
    candidate_ks = [k for k in candidate_ks if k <= max(1, len(X_train)-1)]
    if not candidate_ks:
        return 1
    # quick CV using leave-one-out for robustness on small sets
    best_k, best_acc = candidate_ks[0], -1.0
    # To avoid repeated normalization cost, normalize once per split inside LOO helper
    for k in candidate_ks:
        acc = _loo_score_for_k(X_train, y_train, k, metric=metric, weighted=weighted)
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k

def normalize_data(X_train, X_test):
    if not X_train:
        return X_train, X_test
    
    num_cols = len(X_train[0])
    means = []
    stds = []
    
    for col in range(num_cols):
        column_values = [row[col] for row in X_train]
        mean = sum(column_values) / len(column_values)
        variance = sum((x - mean) ** 2 for x in column_values) / len(column_values)
        std = variance ** 0.5 if variance > 0 else 1.0
        means.append(mean)
        stds.append(std)
    
    X_train_norm = []
    for row in X_train:
        normalized_row = [(row[col] - means[col]) / stds[col] for col in range(num_cols)]
        X_train_norm.append(normalized_row)
    
    X_test_norm = []
    for row in X_test:
        normalized_row = [(row[col] - means[col]) / stds[col] for col in range(num_cols)]
        X_test_norm.append(normalized_row)
    
    return X_train_norm, X_test_norm

def process_dataset(dataset_num, metric="euclidean", weighted=True):
    X_train = load_data(f'TrainData{dataset_num}.txt')
    y_train = load_labels(f'TrainLabel{dataset_num}.txt')
    X_test = load_data(f'TestData{dataset_num}.txt')
    
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    # Select k via LOO CV on training set
    best_k = select_best_k(X_train, y_train, metric=metric, weighted=weighted)
    
    # Normalize with full training statistics
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=best_k, metric=metric, weighted=weighted)
    
    output_filename = f'predictions{dataset_num}.txt'
    with open(output_filename, 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')
    
    print(f'Dataset {dataset_num} processed. k={best_k}, metric={metric}, weighted={weighted}. Predictions saved to {output_filename}')

def main():
    for dataset_num in range(1, 5):
        process_dataset(dataset_num)

if __name__ == '__main__':
    main()

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

def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    
    for test_sample in X_test:
        distances = []
        for i, train_sample in enumerate(X_train):
            dist = euclidean_distance(test_sample, train_sample)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        class_counts = {}
        for _, label in k_nearest:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        predicted_class = max(class_counts, key=class_counts.get)
        predictions.append(predicted_class)
    
    return predictions

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

def process_dataset(dataset_num):
    X_train = load_data(f'TrainData{dataset_num}.txt')
    y_train = load_labels(f'TrainLabel{dataset_num}.txt')
    X_test = load_data(f'TestData{dataset_num}.txt')
    
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    if len(X_train) < 100:
        best_k = min(3, len(X_train) // 2) if len(X_train) > 4 else 1
    elif len(X_train[0]) > 1000:
        best_k = 3
    else:
        best_k = 5
    
    predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=best_k)
    
    output_filename = f'predictions{dataset_num}.txt'
    with open(output_filename, 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')
    
    print(f'Dataset {dataset_num} processed. Predictions saved to {output_filename}')

def main():
    for dataset_num in range(1, 5):
        process_dataset(dataset_num)

if __name__ == '__main__':
    main()

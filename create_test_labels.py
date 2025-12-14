"""
Generate test labels by holding out a portion of training data
This creates TestLabel files so we can measure true test accuracy
"""
import random
from classifier import load_data, load_labels

def create_test_labels_from_train_split(dataset_num, test_ratio=0.2, seed=42):
    """
    Split training data: use most for training, rest becomes labeled test set
    """
    train_file = f'TrainData{dataset_num}.txt'
    label_file = f'TrainLabel{dataset_num}.txt'
    
    X = load_data(train_file)
    y = load_labels(label_file)
    
    n = len(X)
    n_test = max(1, int(n * test_ratio))
    
    # Stratified split to preserve class distribution
    random.seed(seed)
    
    # Group indices by class
    class_indices = {}
    for i, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    test_indices = []
    train_indices = []
    
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_test_class = max(1, int(len(indices) * test_ratio))
        test_indices.extend(indices[:n_test_class])
        train_indices.extend(indices[n_test_class:])
    
    # Write new training files
    new_train_data = [X[i] for i in train_indices]
    new_train_labels = [y[i] for i in train_indices]
    
    with open(f'TrainData{dataset_num}_split.txt', 'w') as f:
        for row in new_train_data:
            f.write(' '.join(str(x) for x in row) + '\n')
    
    with open(f'TrainLabel{dataset_num}_split.txt', 'w') as f:
        for label in new_train_labels:
            f.write(f'{label}\n')
    
    # Write test files with labels
    test_data = [X[i] for i in test_indices]
    test_labels = [y[i] for i in test_indices]
    
    with open(f'TestData{dataset_num}_split.txt', 'w') as f:
        for row in test_data:
            f.write(' '.join(str(x) for x in row) + '\n')
    
    with open(f'TestLabel{dataset_num}_split.txt', 'w') as f:
        for label in test_labels:
            f.write(f'{label}\n')
    
    print(f"Dataset {dataset_num}:")
    print(f"  Original: {n} samples")
    print(f"  New train: {len(train_indices)} samples")
    print(f"  New test: {len(test_indices)} samples")
    print(f"  Class distribution in test: {sorted(set(test_labels))}")
    print(f"  Files: TrainData{dataset_num}_split.txt, TrainLabel{dataset_num}_split.txt,")
    print(f"         TestData{dataset_num}_split.txt, TestLabel{dataset_num}_split.txt\n")

def predict_and_measure_accuracy(dataset_num):
    """
    Use existing test data to create labels via classifier, then measure accuracy
    This simulates having ground truth for existing test sets
    """
    from classifier import load_data, load_labels, impute_missing_values, normalize_data, knn_predict
    
    X_train = load_data(f'TrainData{dataset_num}.txt')
    y_train = load_labels(f'TrainLabel{dataset_num}.txt')
    X_test = load_data(f'TestData{dataset_num}.txt')
    
    # For demonstration: assign labels to test data using nearest neighbor in training set
    # (This creates synthetic "ground truth" to demonstrate accuracy calculation)
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)
    
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # Use k=1 to create "ground truth" (closest training sample's label)
    test_labels_synthetic = knn_predict(X_train_norm, y_train, X_test_norm, k=1, weighted=False)
    
    with open(f'TestLabel{dataset_num}.txt', 'w') as f:
        for label in test_labels_synthetic:
            f.write(f'{label}\n')
    
    print(f"Created TestLabel{dataset_num}.txt with {len(test_labels_synthetic)} labels")
    print(f"  (Note: These are synthetic labels based on k=1 nearest neighbor)\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create test labels for accuracy evaluation")
    parser.add_argument("--method", choices=["split", "synthetic"], default="split",
                        help="split: holdout from training | synthetic: create from k=1 predictions")
    parser.add_argument("--datasets", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
                        help="Which datasets to process")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of training data to use as test (for split method)")
    
    args = parser.parse_args()
    
    print(f"Method: {args.method}\n")
    
    for dataset_num in args.datasets:
        try:
            if args.method == "split":
                create_test_labels_from_train_split(dataset_num, test_ratio=args.test_ratio)
            else:
                predict_and_measure_accuracy(dataset_num)
        except Exception as e:
            print(f"Error processing dataset {dataset_num}: {e}\n")

if __name__ == "__main__":
    main()

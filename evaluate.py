import os
from typing import List, Tuple
from classifier import load_data, load_labels, impute_missing_values, normalize_data, knn_predict


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def evaluate_with_test_labels(dataset_num: int, k: int) -> Tuple[float, int]:
    test_data = f"TestData{dataset_num}.txt"
    test_labels = f"TestLabel{dataset_num}.txt"
    train_data = f"TrainData{dataset_num}.txt"
    train_labels = f"TrainLabel{dataset_num}.txt"

    if not all(os.path.exists(p) for p in [test_data, test_labels, train_data, train_labels]):
        raise FileNotFoundError("Required files not found for labeled test evaluation.")

    X_train = load_data(train_data)
    y_train = load_labels(train_labels)
    X_test = load_data(test_data)
    y_test = load_labels(test_labels)

    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)

    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    y_pred = knn_predict(X_train_norm, y_train, X_test_norm, k=k)

    return accuracy_score(y_test, y_pred), len(y_test)


def leave_one_out_accuracy(dataset_num: int, k: int) -> Tuple[float, int]:
    train_data = f"TrainData{dataset_num}.txt"
    train_labels = f"TrainLabel{dataset_num}.txt"

    if not all(os.path.exists(p) for p in [train_data, train_labels]):
        raise FileNotFoundError("Training files not found for LOO evaluation.")

    X = load_data(train_data)
    y = load_labels(train_labels)

    X = impute_missing_values(X)

    n = len(X)
    if n == 0:
        return 0.0, 0

    correct = 0
    for i in range(n):
        X_train = [row for j, row in enumerate(X) if j != i]
        y_train = [lab for j, lab in enumerate(y) if j != i]
        X_test = [X[i]]

        # Normalize based on training split
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        pred = knn_predict(X_train_norm, y_train, X_test_norm, k=k)[0]
        if pred == y[i]:
            correct += 1

    return correct / n, n


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate accuracy for ML datasets")
    parser.add_argument("--dataset", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Dataset number (1-4)")
    parser.add_argument("--k", type=int, default=5, help="K value for k-NN")
    parser.add_argument("--use-test-labels", action="store_true",
                        help="If TestLabel*.txt exists, evaluate on test labels; else LOO on training")

    args = parser.parse_args()

    if args.use_test_labels:
        try:
            acc, n = evaluate_with_test_labels(args.dataset, args.k)
            print(f"Test accuracy (dataset {args.dataset}, n={n}, k={args.k}): {acc:.4f}")
            return
        except FileNotFoundError:
            print("Test labels not found; falling back to leave-one-out on training.")

    acc, n = leave_one_out_accuracy(args.dataset, args.k)
    print(f"LOO training accuracy (dataset {args.dataset}, n={n}, k={args.k}): {acc:.4f}")


if __name__ == "__main__":
    main()

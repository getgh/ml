Similar Project as inspiration: https://github.com/hurkanugur/SMS-Spam-Classifier/

## 1. Classification

**Classification** is the task of identifying which category a new observation belongs to, based on a provided training dataset. For each of the five datasets, you are given training data, training labels, and test data. Use the training data and labels to build a classifier and predict the labels for the test data. Class labels are integers (e.g., in Dataset 1: 1, 2, 3, 4, 5).

---

## Missing Values
Some datasets contain missing values represented as `1.00000000000000e+99`. These must be imputed before running your classification algorithm.

---

## Dataset Descriptions

### Dataset 1
- Training data: 3312 features × 150 samples  
- Test data: 3312 features × 53 samples  
- Classes: 5  

### Dataset 2
- Training data: 9182 features × 100 samples  
- Test data: 9182 features × 74 samples  
- Classes: 11  

### Dataset 3
- Training data: 112 features × 2547 samples  
- Test data: 112 features × 1092 samples  
- Classes: 9  

### Dataset 4
- Training data: 11 features × 1119 samples  
- Test data: 11 features × 480 samples  
- Classes: 6  

---

## Sample Data

### Training Data
1.1  2.1  2.1  5.2  
2.1  2.4  2.4  2.1  
3.1  1.5  2.6  1.5  

### Training Labels
1  
1  
2  

### Test Data
3.1  2.2  1.5  2.5  
2.1  2.1  2.1  2.6  

---

## Example Prediction Output
If the classifier predicts:

- First test sample → 1  
- Second test sample → 2  

Then the output should be:

1  
2  

---

## Deliverables
Return **one output file per dataset**, each containing the predicted class labels for the test samples, with one label per line.

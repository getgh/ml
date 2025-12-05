# **How to Use the ML Classification Project**

## **Prerequisites**

Before running this project, ensure you have:
- Python 3.7 or higher installed
- pip (Python package manager)
- Git (for cloning the repository)

---

## **Installation & Setup**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/getgh/ml.git
cd ml
```

### **Step 2: Create a Virtual Environment (Recommended)**
```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `gradio` - Web interface framework
- `numpy` - Numerical computing (if needed)

If `requirements.txt` doesn't exist, install manually:
```bash
pip install gradio
```

---

## **Running the Project**

### **Option 1: Batch Processing (Command Line)**

Process all 4 datasets automatically and generate predictions:

```bash
python classifier.py
```

**What happens:**
- Loads training data and labels for all 4 datasets
- Applies data preprocessing (imputation + normalization)
- Trains k-NN classifier on each dataset
- Generates predictions for test data
- Creates output files: `predictions1.txt`, `predictions2.txt`, `predictions3.txt`, `predictions4.txt`

**Expected Output:**
```
Dataset 1 processed. Predictions saved to predictions1.txt
Dataset 2 processed. Predictions saved to predictions2.txt
Dataset 3 processed. Predictions saved to predictions3.txt
Dataset 4 processed. Predictions saved to predictions4.txt
```

---

### **Option 2: Interactive Web Interface (Recommended)**

Launch the Gradio web interface for interactive classification:

```bash
python gradio_app.py
```

**What happens:**
- Opens a web server at `http://localhost:7860`
- Provides an interactive interface with multiple tabs

**Features:**
1. **📁 Existing Datasets Tab**
   - Select dataset (1-4)
   - Adjust k value (1-20)
   - Click "Classify"
   - View results in real-time

2. **📤 Upload Custom Data Tab**
   - Upload your own training data (`.txt`)
   - Upload training labels (`.txt`)
   - Upload test data (`.txt`)
   - Adjust k value
   - Get predictions immediately

3. **ℹ️ About Tab**
   - Algorithm documentation
   - File format requirements
   - Usage instructions

**To stop the server:**
Press `Ctrl + C` in the terminal

---

## **File Structure**

```
ml/
├── classifier.py                 # Core k-NN algorithm
├── gradio_app.py                 # Web interface
├── .gitignore                    # Git ignore file
├── PRESENTATION.md               # Presentation guide
├── README.md                     # This file
│
├── TrainData1.txt               # Dataset 1 - Training features (150 samples × 3312 features)
├── TrainLabel1.txt              # Dataset 1 - Training labels (150 classes)
├── TestData1.txt                # Dataset 1 - Test features (53 samples × 3312 features)
├── predictions1.txt             # Dataset 1 - Predicted classes (53 predictions)
│
├── TrainData2.txt               # Dataset 2 - Training features (100 samples × 9182 features)
├── TrainLabel2.txt              # Dataset 2 - Training labels
├── TestData2.txt                # Dataset 2 - Test features
├── predictions2.txt             # Dataset 2 - Predictions
│
├── TrainData3.txt               # Dataset 3 - Training features (2547 samples × 112 features)
├── TrainLabel3.txt              # Dataset 3 - Training labels
├── TestData3.txt                # Dataset 3 - Test features
├── predictions3.txt             # Dataset 3 - Predictions
│
├── TrainData4.txt               # Dataset 4 - Training features (1119 samples × 11 features)
├── TrainLabel4.txt              # Dataset 4 - Training labels
├── TestData4.txt                # Dataset 4 - Test features
├── predictions4.txt             # Dataset 4 - Predictions
│
├── project-summ.md              # Project summary
└── task-inspiration.md          # Project background
```

---

## **Input File Format**

### **Training Data Format** (`TrainData#.txt`)
Space-separated numerical values, one sample per line:
```
1.5 2.3 1.8 ... 3.2 2.1
2.1 3.4 2.9 ... 1.5 3.0
3.2 1.9 2.5 ... 2.8 1.7
```

### **Training Labels Format** (`TrainLabel#.txt`)
One integer class label per line:
```
1
2
1
```

### **Test Data Format** (`TestData#.txt`)
Same format as training data (space-separated values, no labels):
```
1.6 2.4 1.9 ... 3.1 2.2
2.2 3.5 3.0 ... 1.6 3.1
```

---

## **Output Format**

### **Predictions Format** (`predictions#.txt`)
One predicted class per line (one per test sample):
```
2
1
1
2
3
```

---

## **Common Use Cases**

### **Use Case 1: Run Batch Predictions on Existing Data**
```bash
python classifier.py
# All 4 datasets are processed automatically
# Results saved to predictions1.txt - predictions4.txt
```

### **Use Case 2: Test with Custom Data**
1. Prepare your data files (training, labels, test)
2. Run: `python gradio_app.py`
3. Go to "📤 Upload Custom Data" tab
4. Upload your files
5. Click "Classify Custom Data"
6. View predictions

### **Use Case 3: Experiment with Different k Values**
```bash
python gradio_app.py
# Use the k slider (1-20) in the interface
# See how different k values affect predictions
```

---

## **Algorithm Parameters**

### **k Value (Number of Neighbors)**
- **Default**: 5
- **Range**: 1-20
- **Effect**: 
  - Lower k (1-3): More sensitive to individual neighbors
  - Higher k (10-20): More stable, considers broader neighborhood
- **How it's chosen**: Automatically adapted based on dataset size

### **Distance Metric**
- **Method**: Euclidean distance
- **Formula**: √((x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²)

---

## **Data Preprocessing Steps**

The algorithm automatically performs:

1. **Missing Value Imputation**
   - Identifies values = 1e+99 (missing data marker)
   - Replaces with column mean

2. **Feature Normalization**
   - Standardizes each feature: (value - mean) / std_dev
   - Prevents features with large scales from dominating distance calculation

---

## **Troubleshooting**

### **Error: "Module not found: gradio"**
```bash
pip install gradio
```

### **Error: "File not found: TrainData1.txt"**
- Ensure all data files are in the same directory as the scripts
- Check file names are exact: `TrainData1.txt`, not `traindata1.txt` (case-sensitive)

### **Error: "Address already in use" when running gradio_app.py**
```bash
# The port 7860 is already in use. Kill the process or use a different port
# In gradio_app.py, change the last line to:
demo.launch(server_name="127.0.0.1", server_port=7861)
```

### **Predictions seem wrong**
- Check that training and test data have the same number of features
- Verify training labels match the number of training samples
- Try different k values

---

## **Performance Tips**

1. **For large datasets**: Use smaller k values to speed up computation
2. **For high-dimensional data**: Consider feature selection or dimensionality reduction
3. **For accuracy**: Experiment with k values (typically 3, 5, 7, 9)

---

## **Next Steps**

1. Run `python classifier.py` to generate initial predictions
2. Launch `python gradio_app.py` to explore interactively
3. Upload your own data to test custom datasets
4. Adjust parameters and observe results
5. Review `PRESENTATION.md` for detailed technical information

---

## **Support**

For questions or issues:
- Check `PRESENTATION.md` for detailed algorithm explanation
- Review `project-summ.md` for project background
- Check file formats in this README

---

**Happy classifying! 🚀**

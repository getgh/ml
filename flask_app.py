from flask import Flask, render_template_string, request, jsonify
import tempfile
import os
from classifier import load_data, load_labels, impute_missing_values, normalize_data, knn_predict

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ML Classification Project</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; background: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; margin: 0 auto; background: white; 
            border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 30px; text-align: center; 
        }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        
        .tabs { display: flex; border-bottom: 1px solid #ddd; }
        .tab { 
            flex: 1; padding: 20px; text-align: center; cursor: pointer; 
            background: #f8f9fa; border: none; font-size: 16px;
            transition: all 0.3s ease;
        }
        .tab:hover { background: #e9ecef; }
        .tab.active { background: white; border-bottom: 3px solid #667eea; }
        
        .tab-content { display: none; padding: 30px; }
        .tab-content.active { display: block; }
        
        .form-group { margin-bottom: 25px; }
        .form-group label { 
            display: block; margin-bottom: 8px; font-weight: 600; 
            color: #333;
        }
        .form-group input, .form-group select { 
            width: 100%; padding: 12px; border: 2px solid #ddd; 
            border-radius: 6px; font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .form-group input:focus, .form-group select:focus { 
            outline: none; border-color: #667eea; 
        }
        
        .button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 15px 30px; border: none; 
            border-radius: 6px; cursor: pointer; font-size: 16px;
            transition: transform 0.2s ease;
        }
        .button:hover { transform: translateY(-2px); }
        
        .result-section { 
            margin-top: 30px; padding: 20px; 
            background: #f8f9fa; border-radius: 6px;
        }
        .result-section h3 { margin-top: 0; color: #333; }
        
        .predictions-box { 
            background: white; border: 1px solid #ddd; 
            border-radius: 6px; padding: 15px; 
            max-height: 300px; overflow-y: auto;
            font-family: 'Courier New', monospace;
        }
        
        .info-section { 
            background: #e7f3ff; padding: 20px; 
            border-radius: 6px; margin-bottom: 20px;
        }
        .info-section h3 { margin-top: 0; color: #0066cc; }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .tabs { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ML Classification Interface</h1>
            <p>k-Nearest Neighbors Classification with Missing Value Imputation</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="openTab('existing')">📁 Existing Datasets</button>
            <button class="tab" onclick="openTab('upload')">📤 Upload Custom Data</button>
            <button class="tab" onclick="openTab('about')">ℹ️ About</button>
        </div>
        
        <!-- Existing Datasets Tab -->
        <div id="existing" class="tab-content active">
            <h2>Classify using pre-loaded datasets (1-4)</h2>
            
            <div class="grid">
                <div>
                    <div class="form-group">
                        <label for="dataset-select">Select Dataset:</label>
                        <select id="dataset-select">
                            <option value="1">Dataset 1 (3312 features × 150 samples, 5 classes)</option>
                            <option value="2">Dataset 2 (9182 features × 100 samples, 11 classes)</option>
                            <option value="3">Dataset 3 (112 features × 2547 samples, 9 classes)</option>
                            <option value="4">Dataset 4 (11 features × 1119 samples, 6 classes)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="k-value">K Value (number of neighbors):</label>
                        <input type="range" id="k-value" min="1" max="20" value="5" oninput="updateKValue(this.value)">
                        <span>K = <strong id="k-display">5</strong></span>
                    </div>
                    
                    <button class="button" onclick="classifyExisting()">🚀 Classify</button>
                </div>
                
                <div id="existing-results" class="result-section" style="display: none;">
                    <h3>Results</h3>
                    <div id="existing-info"></div>
                    <h4>Predictions:</h4>
                    <div id="existing-predictions" class="predictions-box"></div>
                </div>
            </div>
        </div>
        
        <!-- Upload Custom Data Tab -->
        <div id="upload" class="tab-content">
            <h2>Upload your own training and test data</h2>
            
            <div class="info-section">
                <h3>📋 File Format Requirements</h3>
                <p><strong>Training Data:</strong> Space-separated numerical features (one sample per line)</p>
                <p><strong>Training Labels:</strong> One integer class label per line</p>
                <p><strong>Test Data:</strong> Same format as training data (without labels)</p>
            </div>
            
            <div class="grid">
                <div>
                    <div class="form-group">
                        <label for="train-file">Training Data (.txt):</label>
                        <input type="file" id="train-file" accept=".txt">
                    </div>
                    
                    <div class="form-group">
                        <label for="label-file">Training Labels (.txt):</label>
                        <input type="file" id="label-file" accept=".txt">
                    </div>
                    
                    <div class="form-group">
                        <label for="test-file">Test Data (.txt):</label>
                        <input type="file" id="test-file" accept=".txt">
                    </div>
                    
                    <div class="form-group">
                        <label for="k-custom">K Value:</label>
                        <input type="range" id="k-custom" min="1" max="20" value="5" oninput="updateKCustom(this.value)">
                        <span>K = <strong id="k-custom-display">5</strong></span>
                    </div>
                    
                    <button class="button" onclick="classifyCustom()">🚀 Classify Custom Data</button>
                </div>
                
                <div id="custom-results" class="result-section" style="display: none;">
                    <h3>Results</h3>
                    <div id="custom-info"></div>
                    <h4>Predictions:</h4>
                    <div id="custom-predictions" class="predictions-box"></div>
                </div>
            </div>
        </div>
        
        <!-- About Tab -->
        <div id="about" class="tab-content">
            <h2>About This Project</h2>
            
            <div class="info-section">
                <h3>🎯 Project Overview</h3>
                <p>This is a <strong>k-Nearest Neighbors (k-NN) classifier</strong> designed for multi-dataset classification tasks with the following capabilities:</p>
                <ul>
                    <li>✅ <strong>Missing value imputation</strong> (replaces 1e+99 with column means)</li>
                    <li>✅ <strong>Data normalization</strong> (standardizes features)</li>
                    <li>✅ <strong>Multi-class classification</strong></li>
                    <li>✅ <strong>Adaptive k-value selection</strong> based on dataset characteristics</li>
                </ul>
            </div>
            
            <div class="info-section">
                <h3>🔬 Algorithm Details</h3>
                <ul>
                    <li><strong>Distance Metric:</strong> Euclidean distance for similarity measurement</li>
                    <li><strong>Missing Values:</strong> Automatically handles values represented as 1e+99</li>
                    <li><strong>Feature Scaling:</strong> Normalizes features to prevent scale bias</li>
                    <li><strong>Classification:</strong> Majority voting among k-nearest neighbors</li>
                </ul>
            </div>
            
            <div class="info-section">
                <h3>🌍 Real-World Applications</h3>
                <ul>
                    <li><strong>Healthcare:</strong> Disease diagnosis, drug effectiveness prediction</li>
                    <li><strong>Finance:</strong> Credit risk assessment, fraud detection</li>
                    <li><strong>Technology:</strong> Image recognition, sentiment analysis</li>
                    <li><strong>Business:</strong> Customer segmentation, recommendation systems</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        function openTab(tabName) {
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => tab.classList.remove('active'));
            contents.forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function updateKValue(value) {
            document.getElementById('k-display').textContent = value;
        }
        
        function updateKCustom(value) {
            document.getElementById('k-custom-display').textContent = value;
        }
        
        async function classifyExisting() {
            const dataset = document.getElementById('dataset-select').value;
            const k = document.getElementById('k-value').value;
            
            try {
                const response = await fetch('/classify_existing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset: parseInt(dataset), k: parseInt(k) })
                });
                
                const result = await response.json();
                
                document.getElementById('existing-info').innerHTML = result.info;
                document.getElementById('existing-predictions').textContent = result.predictions;
                document.getElementById('existing-results').style.display = 'block';
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function classifyCustom() {
            const trainFile = document.getElementById('train-file').files[0];
            const labelFile = document.getElementById('label-file').files[0];
            const testFile = document.getElementById('test-file').files[0];
            const k = document.getElementById('k-custom').value;
            
            if (!trainFile || !labelFile || !testFile) {
                alert('Please upload all three files');
                return;
            }
            
            const formData = new FormData();
            formData.append('train_file', trainFile);
            formData.append('label_file', labelFile);
            formData.append('test_file', testFile);
            formData.append('k', k);
            
            try {
                const response = await fetch('/classify_custom', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                document.getElementById('custom-info').innerHTML = result.info;
                document.getElementById('custom-predictions').textContent = result.predictions;
                document.getElementById('custom-results').style.display = 'block';
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/classify_existing', methods=['POST'])
def classify_existing():
    try:
        data = request.json
        dataset_num = data['dataset']
        k_value = data['k']
        
        train_file = f'TrainData{dataset_num}.txt'
        label_file = f'TrainLabel{dataset_num}.txt'
        test_file = f'TestData{dataset_num}.txt'
        
        if not all(os.path.exists(f) for f in [train_file, label_file, test_file]):
            return jsonify({'error': f'Dataset {dataset_num} files not found'})
        
        X_train = load_data(train_file)
        y_train = load_labels(label_file)
        X_test = load_data(test_file)
        
        X_train = impute_missing_values(X_train)
        X_test = impute_missing_values(X_test)
        
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        
        predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=k_value)
        
        info = f"""
        <p><strong>Dataset {dataset_num} Classification Results:</strong></p>
        <ul>
            <li>Training samples: {len(X_train)}</li>
            <li>Features: {len(X_train[0])}</li>
            <li>Test samples: {len(X_test)}</li>
            <li>Classes: {sorted(set(y_train))}</li>
            <li>K value used: {k_value}</li>
        </ul>
        """
        
        predictions_text = '\n'.join(str(pred) for pred in predictions)
        
        return jsonify({
            'info': info,
            'predictions': predictions_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/classify_custom', methods=['POST'])
def classify_custom():
    try:
        train_file = request.files['train_file']
        label_file = request.files['label_file']
        test_file = request.files['test_file']
        k_value = int(request.form['k'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = os.path.join(temp_dir, "train.txt")
            label_path = os.path.join(temp_dir, "labels.txt")
            test_path = os.path.join(temp_dir, "test.txt")
            
            train_file.save(train_path)
            label_file.save(label_path)
            test_file.save(test_path)
            
            X_train = load_data(train_path)
            y_train = load_labels(label_path)
            X_test = load_data(test_path)
            
            if not X_train or not y_train or not X_test:
                return jsonify({'error': 'Could not load data from files'})
            
            if len(X_train) != len(y_train):
                return jsonify({'error': f'Training data ({len(X_train)} samples) and labels ({len(y_train)} labels) don\'t match'})
            
            X_train = impute_missing_values(X_train)
            X_test = impute_missing_values(X_test)
            
            X_train_norm, X_test_norm = normalize_data(X_train, X_test)
            
            predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=k_value)
            
            info = f"""
            <p><strong>Custom Data Classification Results:</strong></p>
            <ul>
                <li>Training samples: {len(X_train)}</li>
                <li>Features: {len(X_train[0])}</li>
                <li>Test samples: {len(X_test)}</li>
                <li>Classes found: {sorted(set(y_train))}</li>
                <li>K value used: {k_value}</li>
            </ul>
            """
            
            predictions_text = '\n'.join(str(pred) for pred in predictions)
            
            return jsonify({
                'info': info,
                'predictions': predictions_text
            })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

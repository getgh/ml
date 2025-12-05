import gradio as gr
import os
from classifier import load_data, load_labels, impute_missing_values, normalize_data, knn_predict

def classify_custom_data(train_file, label_file, test_file, k_value):
    try:
        if not all([train_file, label_file, test_file]):
            return "Please upload all three files (training data, labels, test data)", ""
        
        train_path = train_file if isinstance(train_file, str) else train_file.name
        label_path = label_file if isinstance(label_file, str) else label_file.name
        test_path = test_file if isinstance(test_file, str) else test_file.name
        
        X_train = load_data(train_path)
        y_train = load_labels(label_path)
        X_test = load_data(test_path)
        
        if not X_train or not y_train or not X_test:
            return "Error: Could not load data from files", ""
        
        if len(X_train) != len(y_train):
            return f"Error: Training data ({len(X_train)} samples) and labels ({len(y_train)} labels) don't match", ""
        
        X_train = impute_missing_values(X_train)
        X_test = impute_missing_values(X_test)
        
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        
        predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=k_value)
        
        result_text = f"Classification completed successfully!\n"
        result_text += f"Training samples: {len(X_train)}\n"
        result_text += f"Features: {len(X_train[0])}\n"
        result_text += f"Test samples: {len(X_test)}\n"
        result_text += f"Classes found: {sorted(set(y_train))}\n"
        result_text += f"K value used: {k_value}"
        
        predictions_text = "\n".join(str(pred) for pred in predictions)
        
        return result_text, predictions_text
    
    except Exception as e:
        return f"Error: {str(e)}", ""

def classify_existing_dataset(dataset_num, k_value):
    try:
        train_file = f'TrainData{dataset_num}.txt'
        label_file = f'TrainLabel{dataset_num}.txt'
        test_file = f'TestData{dataset_num}.txt'
        
        if not all(os.path.exists(f) for f in [train_file, label_file, test_file]):
            return f"Dataset {dataset_num} files not found", ""
        
        X_train = load_data(train_file)
        y_train = load_labels(label_file)
        X_test = load_data(test_file)
        
        X_train = impute_missing_values(X_train)
        X_test = impute_missing_values(X_test)
        
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        
        predictions = knn_predict(X_train_norm, y_train, X_test_norm, k=k_value)
        
        result_text = f"Dataset {dataset_num} Classification Results:\n"
        result_text += f"Training samples: {len(X_train)}\n"
        result_text += f"Features: {len(X_train[0])}\n"
        result_text += f"Test samples: {len(X_test)}\n"
        result_text += f"Classes: {sorted(set(y_train))}\n"
        result_text += f"K value used: {k_value}"
        
        predictions_text = "\n".join(str(pred) for pred in predictions)
        
        return result_text, predictions_text
    
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks(title="ML Classification Project") as demo:
    gr.Markdown("# Machine Learning Classification Interface")
    gr.Markdown("This interface allows you to run k-NN classification on existing datasets or upload your own data.")
    
    with gr.Tabs():
        with gr.TabItem("📁 Existing Datasets"):
            gr.Markdown("### Classify using pre-loaded datasets (1-4)")
            
            with gr.Row():
                dataset_dropdown = gr.Dropdown(
                    choices=[1, 2, 3, 4],
                    label="Select Dataset",
                    value=1
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                    label="K Value (number of neighbors)"
                )
            
            classify_btn = gr.Button("Classify", variant="primary")
            
            with gr.Row():
                result_text = gr.Textbox(
                    label="Results",
                    lines=6,
                    interactive=False
                )
                predictions_text = gr.Textbox(
                    label="Predictions (one per line)",
                    lines=10,
                    interactive=False
                )
            
            classify_btn.click(
                classify_existing_dataset,
                inputs=[dataset_dropdown, k_slider],
                outputs=[result_text, predictions_text]
            )
        
        with gr.TabItem("Upload Custom Data"):
            gr.Markdown("### Upload your own training and test data")
            gr.Markdown("**File Format:** Text files with space-separated values")
            
            with gr.Row():
                with gr.Column():
                    train_upload = gr.File(
                        label="Training Data (.txt)",
                        file_types=[".txt"]
                    )
                    label_upload = gr.File(
                        label="Training Labels (.txt)", 
                        file_types=[".txt"]
                    )
                    test_upload = gr.File(
                        label="Test Data (.txt)",
                        file_types=[".txt"]
                    )
                
                with gr.Column():
                    k_custom = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                        label="K Value"
                    )
            
            upload_classify_btn = gr.Button("Classify Custom Data", variant="primary")
            
            with gr.Row():
                custom_result = gr.Textbox(
                    label="Results",
                    lines=6,
                    interactive=False
                )
                custom_predictions = gr.Textbox(
                    label="Predictions",
                    lines=10,
                    interactive=False
                )
            
            upload_classify_btn.click(
                classify_custom_data,
                inputs=[train_upload, label_upload, test_upload, k_custom],
                outputs=[custom_result, custom_predictions]
            )
        
        with gr.TabItem("About"):
            gr.Markdown("""
            ###            
            This is a **k-Nearest Neighbors (k-NN) classifier** that can handle:
            
            -  **Missing value imputation** (replaces 1e+99 with column means)
            -  **Data normalization** (standardizes features)
            -  **Multi-class classification** 
            -  **Adaptive k-value selection** based on dataset characteristics
            
            ### Dataset Requirements
            
            **Training Data:** Space-separated numerical features (one sample per line)
            ```
            1.1 2.1 2.1 5.2
            2.1 2.4 2.4 2.1
            3.1 1.5 2.6 1.5
            ```
            
            **Training Labels:** One integer class label per line
            ```
            1
            1
            2
            ```
            
            **Test Data:** Same format as training data (without labels)
            
            ### Algorithm Details
            - Uses Euclidean distance for similarity measurement
            - Automatically handles missing values (1e+99)
            - Normalizes features to prevent scale bias
            - Outputs one prediction per test sample
            """)

if __name__ == "__main__":
    demo.launch()

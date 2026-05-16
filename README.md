# 🔍 Spam Detection Lab: Transformer-Based Text Classification

A high-performance, multi-model spam detection system powered by **Transformer Encoders**. This project features a custom-built Transformer architecture implemented in PyTorch, trained on diverse datasets (SMS, Enron Email, and Combined sets), and served through a modern, dark-themed Streamlit dashboard.

---

## 🚀 Key Features

-   **Multi-Model Intelligence**: Simultaneously run and compare predictions from three distinct models:
    -   `sms_model`: Optimized for short-form SMS spam.
    -   `youtube_model`: Trained on the Enron dataset (renamed for specific lab context).
    -   `reviews_model`: Trained on a broad combined dataset.
-   **Custom Transformer Architecture**: Built from scratch using `nn.TransformerEncoderLayer` for efficient sequential dependency capture.
-   **Advanced Dashboard**: A professional Streamlit UI with:
    -   **Real-time Inference**: Analyze any message instantly.
    -   **Artefact Viewer**: Inspect training loss/accuracy curves and confusion matrices directly.
    -   **Performance Metrics**: Live view of F1-scores, accuracy, and dataset distribution.
-   **Automated Pipeline**: End-to-end training script that handles cleaning, vocabulary building, training, and artefact export.

---

## 🛠️ Technical Architecture

### 🧠 Model Specifications
The core is a **Transformer Classifier** designed for text classification:
-   **Embedding Dimension**: 128
-   **Attention Heads**: 4
-   **Encoder Layers**: 3
-   **Max Sequence Length**: 60 tokens
-   **Vocab Size**: ~12,000 tokens (dynamic per dataset)
-   **Optimizer**: Adam (`lr=2e-4`)
-   **Loss Function**: Cross-Entropy Loss

```python
# Model Architecture Overview
TransformerClassifier(
  (embedding): Embedding(vocab_size, 128)
  (transformer): TransformerEncoder(
    (layers): ModuleList(
      (0-2): 3 x TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.3)
    )
  )
  (fc): Linear(in_features=128, out_features=2)
)
```

### 📊 Training Pipeline
1.  **Preprocessing**: Text cleaning (lowercase, punctuation removal, digit removal).
2.  **Tokenization**: Custom frequency-based vocabulary building.
3.  **Dataset**: PyTorch `DataLoader` with 80/20 train-test split.
4.  **Evaluation**: Real-time validation accuracy tracking and automated report generation (`classification_report`).

---

## 📁 Project Structure

```text
.
├── app.py                      # Streamlit dashboard & Inference logic
├── sms_spam_detection.ipynb    # Training pipeline & Model definition
├── Dataset/                    # Raw CSV data
│   ├── combined_dataset.csv
│   ├── enron_spam_data.csv
│   └── spam.csv
├── Saved_model/                # Trained PyTorch checkpoints (.pt)
│   ├── sms_model.pt
│   ├── youtube_model.pt
│   └── reviews_model.pt
├── results/                    # Training artefacts
│   ├── *_accuracy.png          # Learning curves
│   ├── *_cm.png                # Confusion matrices
│   ├── *_report.txt            # Precision/Recall reports
│   └── *_vocab.pkl             # Serialized vocabularies
└── requirements.txt            # Dependency list
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sms-detector-transformer.git
cd sms-detector-transformer
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install torch pandas streamlit matplotlib seaborn scikit-learn
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📈 Model Performance

Based on the training reports, the models achieve high reliability across different domains:

| Model | Accuracy | Target Domain |
| :--- | :--- | :--- |
| **SMS** | ~93.8% | Mobile Text Messages |
| **YouTube** | ~97.2% | Email/Enron Dataset |
| **Reviews** | ~96.1% | Combined General Text |

*Detailed metrics (Precision, Recall, F1) can be viewed in the `results/` folder or via the Dashboard's "Show Panels" feature.*

---

## 🛠️ Usage

### Training New Models
Open `sms_spam_detection.ipynb` in Jupyter or VS Code and run the `run_pipeline` function:
```python
run_pipeline("path/to/data.csv", "text_column", "label_column", "model_name")
```
This will automatically save the model to `Saved_model/` and all plots/vocab to `results/`.

### Testing Messages
1. Launch the Streamlit app.
2. Select your preferred model from the sidebar.
3. Type a message or choose an example from the dropdown.
4. Click **Analyze message** to see probabilistic results and confidence scores.

---

## 🛡️ License
This project is developed for educational purposes as part of the AI-based Spam Detection Lab.

## 👥 Contributors
- **Romio** - Lead Developer & Data Scientist

---
*Built with ❤️ using PyTorch and Streamlit.*

# 📱 SMS Spam Detection — Faster, Safer, Smarter

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)

A professional, end-to-end machine learning solution for detecting SMS spam. This project features a robust TF-IDF + Linear SVC pipeline with a polished Streamlit dashboard for real-time inference and data visualization.

---

## 🚀 Key Features

- **Real-time Classification:** Instantly detect if a message is `SPAM` or `HAM` with confidence scoring.
- **Advanced NLP Pipeline:** Utilizes NLTK for text preprocessing (cleaning, normalization) and Scikit-Learn for feature extraction (TF-IDF).
- **Interactive Dashboard:** A modern UI built with Streamlit featuring:
  - Custom CSS animations and "shimmer" effects.
  - Dataset statistics and distribution metrics.
  - Explanable AI details (showing cleaned text and probability).
  - Pre-loaded example messages for testing.
- **Comprehensive Analysis:** The project includes a Jupyter notebook for EDA, model benchmarking, and evaluation.
- **High Performance:** Achieved **96.4% Accuracy** and **0.91 F1-Score** using Linear SVC.

---

## 📊 Model Performance

After benchmarking multiple models (Logistic Regression, Naive Bayes, Linear SVC, and Random Forest), **Linear SVC** was selected as the champion model.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 96.41% |
| **F1-Score** | 0.9146 |
| **Dataset Size** | 10,178 examples |
| **Spam/Ham Ratio** | 22% / 78% |

> **Visualizations:** Detailed plots for ROC Curves, Confusion Matrices, and EDA are available in the [`results/`](./results/) directory.

---

## 📂 Project Structure

```text
SpamDetection/
├── app.py                # Main Streamlit application
├── sms_spam_detection.ipynb # Research, EDA, and Model Training notebook
├── requirements.txt      # Python dependencies
├── run.bat               # Shortcut to launch the app
├── Dataset/              # Raw data files
│   ├── spam.csv          # Primary SMS dataset
│   ├── combined_dataset.csv
│   └── enron_spam_data.csv
├── Saved_model/          # Serialized model artifacts
│   ├── spam_pipeline.pkl # Trained TF-IDF + Classifier pipeline
│   └── label_encoder.pkl # Label encoder for spam/ham
├── results/              # Evaluation plots and charts
│   ├── confusion_matrices.png
│   ├── eda_plot.png
│   └── roc.png
└── README.md             # This file
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
The application requires `stopwords` and `punkt` from NLTK. These are automatically handled in the notebook, but you can manually download them:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🎮 How to Run

### Option A: Run the Streamlit App
Launch the interactive dashboard to test your own messages:
```bash
streamlit run app.py
```
Or simply run the provided batch file (Windows):
```bash
run.bat
```

### Option B: Explore the Research
Open the Jupyter notebook to see the data analysis, model comparison, and training process:
```bash
jupyter notebook sms_spam_detection.ipynb
```

---

## 🧠 Technical Workflow

1.  **Preprocessing:**
    - Text normalization (lowercase).
    - Removal of URLs, special characters, and punctuation.
    - Tokenization and stop-word removal.
2.  **Vectorization:**
    - TF-IDF (Term Frequency-Inverse Document Frequency) transforms text into a high-dimensional sparse matrix.
3.  **Classification:**
    - Linear SVC (Support Vector Classification) predicts the final label.
4.  **Deployment:**
    - The pipeline is serialized using `joblib` for efficient loading in the Streamlit app.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

**Built with ❤️ for NLP enthusiasts.**

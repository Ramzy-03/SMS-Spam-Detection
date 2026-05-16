# SMS Spam Detection

## Overview
This project provides a professional Streamlit app for SMS spam detection using a TF-IDF + Logistic Regression pipeline.

## Files
- `app.py`: Updated Streamlit dashboard with clean UX, dataset stats, and animated polished visuals.
- `train_model.py`: Training script that loads `Dataset/spam.csv`, trains the classifier, and saves model artifacts.
- `Dataset/spam.csv`: SMS spam dataset.
- `Saved_model/`: Generated model assets after running training.
- `requirements.txt`: Python dependencies.

## Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model if it is not already available:

```bash
python train_model.py
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes
- The app checks both `Saved_model/` and `model/` for existing pipeline files.
- If the model is missing, the app shows an error and instructs you to run the training script.
- The app now includes dataset preview cards, confidence scoring, and polished Streamlit styling.
"# SMS-Spam-Detection" 

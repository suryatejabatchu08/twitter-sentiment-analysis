# Sentiment Analysis Project

A machine learning project for sentiment analysis using Python.

## Project Structure

- `sentiment_analysis.ipynb`: Jupyter notebook containing the main analysis and model development
- `app.py`: Python application for sentiment prediction
- `log_model.pkl`: Trained Logistic Regression model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer for text preprocessing

### Dataset

The dataset is located in the `Dataset` folder:
- `train.csv`: Training dataset
- `test.csv`: Test dataset

## Setup

1. Create a virtual environment:
```bash
python -m venv tsa
```

2. Activate the virtual environment:
- Windows:
```bash
.\tsa\Scripts\activate
```
- Unix/MacOS:
```bash
source tsa/Scripts/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. To run the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis.ipynb
```

2. To run the application:
```bash
streamlit run app.py
```

## Requirements

See `requirements.txt` for a list of Python dependencies.

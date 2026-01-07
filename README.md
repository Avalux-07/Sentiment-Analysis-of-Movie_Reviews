🎬 Sentiment Analysis of Movie Reviews

A machine learning project that performs sentiment classification (positive / negative) on movie reviews using TF-IDF feature extraction and Support Vector Machine (SVM) classifiers.
The project follows a modular Python structure, supports multiple kernels, and includes model comparison and visualization.

📌 Project Overview

This project aims to:

Load and preprocess raw movie review text data

Convert text into numerical features using TF-IDF

Train multiple SVM models with different kernels

Evaluate models using Accuracy, F1-Score, Confusion Matrix

Compare and visualize model performance

Save trained models and evaluation results for reuse

The dataset is structured into train and test folders with pos and neg subfolders, containing individual .txt review files.

📂 Dataset Structure
clean_dataset/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/


Each file contains one movie review.

🧠 Models Used

Support Vector Machine (SVM)

Linear Kernel

Polynomial Kernel

RBF Kernel

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Kernel-wise performance comparison

📁 Project Structure
Sentiment-Analysis-of-Movie_Reviews/
├── src/
│   ├── __init__.py
│   ├── data_loading_preprocessing.py
│   ├── train.py
│   └── compare_and_visualize.py
├── clean_dataset/
├── results/
│   ├── evaluation_metrics.csv
│   └── kernel_comparison.png
├── saved_models/        # ignored by git
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/Avalux-07/Sentiment-Analysis-of-Movie_Reviews.git
cd Sentiment-Analysis-of-Movie_Reviews

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🚀 Usage
▶ Train models
python -m src.train


This will:

Load and preprocess data

Train SVM models with different kernels

Save trained models and evaluation metrics

▶ Compare results & visualize
python -m src.compare_and_visualize


This will:

Load saved evaluation results

Generate comparison tables

Create visual plots of model performance

📈 Sample Results
Kernel	Accuracy	F1-Score
Linear	0.8722	0.8711
Polynomial	0.8216	0.8255
RBF	0.8789	0.8785

📌 RBF kernel achieved the best overall performance on this dataset.

🧪 Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Joblib

📌 Key Features

Modular and reusable codebase

Clean dataset handling (no CSV dependency)

Model persistence using joblib

Clear evaluation & visualization

GitHub-ready project structure

🔒 Git Ignore Policy

The following are excluded from version control:

Trained models (.pkl)

saved_models/

Cache files (__pycache__/)

Virtual environments

👤 Author

Soumik Debnath
Data Science Student
IIT Guwahati

GitHub: https://github.com/Avalux-07

⭐ Future Improvements

Add deep learning models (LSTM / BERT)

Hyperparameter tuning with GridSearchCV

Cross-validation

Web interface for live predictions

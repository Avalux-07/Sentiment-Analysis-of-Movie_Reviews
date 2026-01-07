# 🎬 Sentiment Analysis of Movie Reviews

A machine learning project that performs **sentiment classification (positive / negative)** on movie reviews using **TF-IDF feature extraction** and **Support Vector Machine (SVM)** classifiers.  
The project follows a **modular Python structure**, supports **multiple kernels**, and includes **model comparison and visualization**.

---

## 📌 Project Overview

This project aims to:
- Load and preprocess raw movie review text data
- Convert text into numerical features using **TF-IDF**
- Train multiple **SVM models** with different kernels
- Evaluate models using **Accuracy, F1-Score, Confusion Matrix**
- Compare and visualize model performance
- Save trained models and evaluation results for reuse

The dataset is structured into `train` and `test` folders with `pos` and `neg` subfolders, containing individual `.txt` review files.

---

## 📂 Dataset Structure
<img width="296" height="211" alt="image" src="https://github.com/user-attachments/assets/2fd5c474-5c25-4bc1-8046-35ecda7d7921" />

---

## 🧠 Models Used

- **Support Vector Machine (SVM)**
  - Linear Kernel
  - Polynomial Kernel
  - RBF Kernel

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Kernel-wise performance comparison

---

## 📁 Project Structure
<img width="636" height="432" alt="image" src="https://github.com/user-attachments/assets/4daeb3cd-1b25-4119-84a6-61bf88b3b5f6" />

---

## ⚙️ Installation

## 1️⃣ Clone the repository
```bash
git clone https://github.com/Avalux-07/Sentiment-Analysis-of-Movie_Reviews.git
cd Sentiment-Analysis-of-Movie_Reviews
```

## 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

## 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

## ▶ Train models
```bash
python -m src.train
```
This will:
1. Load and preprocess data.
2. Train SVM models with different kernels.
3. Save trained models and evaluation metrics.

## ▶ Compare results & visualize
```bash
python -m src.compare_and_visualize
```
This will:
1. Load saved evaluation results.
2. Generate comparison tables.
3. Create visual plots of model performance.

---

## 📈 Sample Results

The performance of different SVM kernels on the movie review dataset is summarized below:

| Kernel       | Accuracy | F1-Score |
|--------------|----------|----------|
| Linear       | 0.87220  | 0.87109  |
| Polynomial   | 0.82164  | 0.82549  |
| RBF          | 0.87888  | 0.87854  |

📌 **Observation:**  
The **RBF kernel** achieved the highest accuracy and F1-score, making it the best-performing model for this task.

---

## 🧪 Technologies Used
1. Python
2. Numpy
3. pandas
4. scikit-learn
5. Matplotlib
6. joblib

---

## 👤 Author
### Soumik Debnath
### Data Science-IIT Guwahati

GitHub: https://github.com/Avalux-07

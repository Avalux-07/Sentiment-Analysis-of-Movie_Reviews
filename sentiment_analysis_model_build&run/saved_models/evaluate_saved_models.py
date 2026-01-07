# Compare results & visualize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import joblib
import os

# Load models and vectorizer from local directory
svm_linear = joblib.load(r"D:\GitHub_uploded\Sentiment-Analysis-of-Movie_Reviews\sentiment_analysis_model_build&run\saved_models\svm_linear.pkl")
svm_poly   = joblib.load(r"D:\GitHub_uploded\Sentiment-Analysis-of-Movie_Reviews\sentiment_analysis_model_build&run\saved_models\svm_poly.pkl")
svm_rbf    = joblib.load(r"D:\GitHub_uploded\Sentiment-Analysis-of-Movie_Reviews\sentiment_analysis_model_build&run\saved_models\svm_rbf.pkl")
tfidf      = joblib.load(r"D:\GitHub_uploded\Sentiment-Analysis-of-Movie_Reviews\sentiment_analysis_model_build&run\saved_models\tfidf_vectorizer.pkl")
print("Models loaded successfully!")

# Load evaluation outputs from local directory
y_test        = joblib.load(r"sentiment_analysis_model_build&run/saved_models/evaluation/y_test.pkl")
y_pred_linear = joblib.load(r"D:\data science\Sentiment_Analysis_Movie_Reviews_Project\saved_models\evaluation\y_pred_linear.pkl")
y_pred_poly   = joblib.load(r"D:\data science\Sentiment_Analysis_Movie_Reviews_Project\saved_models\evaluation\y_pred_poly.pkl")
y_pred_rbf    = joblib.load(r"D:\data science\Sentiment_Analysis_Movie_Reviews_Project\saved_models\evaluation\y_pred_rbf.pkl")

print("Evaluation outputs loaded successfully!")

# Compare models performance
comparison_df = pd.DataFrame({
    "Model": ["Linear", "Polynomial", "RBF"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_linear),
        accuracy_score(y_test, y_pred_poly),
        accuracy_score(y_test, y_pred_rbf)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_linear),
        f1_score(y_test, y_pred_poly),
        f1_score(y_test, y_pred_rbf)
    ]
})
print(comparison_df)

# visualize comparison
comparison_df.set_index("Model").plot.bar(rot=0)
plt.title("Model Comparison on Movie Review Sentiment Analysis")
plt.ylabel("Score")
plt.ylim(0.8, 0.9)
plt.show()

# Confusion Matrix for Linear SVM
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix - SVM RBF")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

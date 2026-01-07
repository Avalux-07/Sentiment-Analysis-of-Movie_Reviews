# train svm linear model

from data_loading_preprocessing import load_and_vectorize_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_vectorize_data()

# Train SVM with linear kernel
svm_linear = SVC(kernel="linear")
svm_linear.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = svm_linear.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



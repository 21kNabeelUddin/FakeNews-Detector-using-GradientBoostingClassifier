# Fake News Detection using GradientBoostingClassifier
This project aims to detect fake news using a machine learning model. The dataset used in this project contains news articles labeled as "Fake" or "Real." The goal of this project is to predict whether a given news article is fake or real based on its text content using the GradientBoostingClassifier algorithm.


## Dataset Description
The dataset contains news articles labeled as either "Fake" or "Real." Each entry consists of the following fields:

Text: The body of the news article.
Title: The headline of the news article.
Label: The label indicating whether the news article is fake (0) or real (1).

## 1. Data Preprocessing
Text Cleaning: Text data is cleaned by removing stop words, punctuation, and special characters. Tokenization is applied, and text is transformed into a format suitable for model training.
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text data into numerical features.
Train-Test Split: The data is split into training and testing sets, with 80% used for training and 20% for testing.

## 2. Model Architecture
Model: The GradientBoostingClassifier algorithm from Scikit-learn is used for classification. It is a powerful ensemble method that combines multiple decision trees to produce accurate predictions.
Hyperparameters: The model is tuned using cross-validation to find the best hyperparameters for optimal performance.

## 3. Model Training
Training: The GradientBoostingClassifier is trained on the preprocessed dataset.
Cross-Validation: K-fold cross-validation is used to evaluate the model's performance and avoid overfitting.
Metrics: The model's performance is evaluated using accuracy, precision, recall, and F1-score.

## 4. Model Evaluation
Accuracy: The accuracy of the model on the test dataset is computed.
Classification Report: A detailed classification report, including precision, recall, and F1-score for each class (Fake and Real), is provided.
Confusion Matrix: A confusion matrix is generated to visualize the performance of the model.

## 5. Testing and Final Evaluation
The trained model is saved and used to predict new news articles. The model is evaluated on the test set to confirm its generalization ability.

Clone the repository:
```bash
git clone https://github.com/21kNabeelUddin/FakeNews-Detector-using-GradientBoostingClassifier.git
```


Install necessary dependencies: Create a virtual environment and install the required libraries using requirements.txt:
```bash
pip install -r requirements.txt
```

move to directory using
```bash
cd FakeNews-Detector-using-GradientBoostingClassifier
```

Run the Jupyter notebook: Open the notebook in Jupyter and execute the code cells step by step.

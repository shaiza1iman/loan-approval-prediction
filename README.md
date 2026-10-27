 Loan Approval Prediction Using Logistic Regression

## Project Overview

This project uses machine learning to predict loan approval status based on applicant information. It was developed as an educational AI/ML internship project.

The model uses Logistic Regression to classify applications as approved (`Y`) or rejected (`N`).

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* Streamlit
* Joblib
* Google Colab

## Dataset

The dataset contains 614 records and 13 columns, including applicant details, financial information, credit history, property area, and loan approval status.

The target column is `Loan_Status`. `Loan_ID` was excluded from model training because it is an identifier.

## Machine Learning Workflow

1. Data exploration
2. Missing-value handling
3. Categorical feature encoding
4. Numerical feature scaling
5. Train/test split
6. Logistic Regression training
7. Model evaluation
8. Streamlit application development

Preprocessing was fitted on the training data and then applied to the test data to help prevent data leakage.

## Model Evaluation

The model was evaluated on a held-out test set of 123 records.

| Metric                             | Result |
| ---------------------------------- | -----: |
| Accuracy                           | 86.18% |
| Precision for approved class (`Y`) | 84.00% |
| Recall for approved class (`Y`)    | 98.82% |
| F1-score for approved class (`Y`)  | 90.81% |
| ROC-AUC                            | 85.23% |

### Confusion Matrix Findings

* True negatives: 22
* False positives: 16
* False negatives: 1
* True positives: 84

The model identified most approved applications in the test set, but incorrectly classified 16 rejected applications as approved. Therefore, accuracy alone does not fully describe model performance.

## Project Structure

```text
loan-approval-prediction/
├── app/
│   └── app.py
├── models/
│   ├── loan_model.pkl
│   └── loan_preprocessor.pkl
├── images/
├── notebooks/
│   └── loan_approval_prediction.ipynb
├── requirements.txt
└── README.md
```

## Run the Application Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your actual repository URL and folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the application

Run from the repository root:

```bash
streamlit run app/app.py
```

The application loads the trained model and fitted preprocessing pipeline from the `models/` directory.

## Application Features

* Collects applicant information through a Streamlit form
* Applies the saved preprocessing pipeline
* Predicts loan approval status
* Displays the model's estimated approval probability

## Limitations and Responsible Use

This is an educational machine learning project, not a real banking or lending decision system.

Predictions and probabilities reflect patterns learned from the available dataset. They are not guarantees of loan approval and should not be used to make real financial decisions.

The model may make incorrect predictions and may reflect limitations or biases present in its training data. Further validation would be required before any real-world application.

## Internship Project

Developed as part of the Big Brain AI/ML internship project.


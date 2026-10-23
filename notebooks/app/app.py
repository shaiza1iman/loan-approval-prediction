import streamlit as st
import pandas as pd
import joblib

# Load the trained model and fitted preprocessor
model = joblib.load("models/loan_model.pkl")
preprocessor = joblib.load("models/loan_preprocessor.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter the applicant information below to generate "
    "a prediction using our Logistic Regression model."
)

st.info(
    "Educational project only. Predictions are not "
    "financial advice or real loan decisions."
)

# Applicant information form
with st.form("loan_application_form"):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox(
            "Dependents", ["0", "1", "2", "3+"]
        )
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed", ["Yes", "No"]
        )

    with col2:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0, value=0
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands)", min_value=1, value=150
        )
        loan_term = st.selectbox(
            "Loan Term (months)", [360, 180, 120, 60, 36, 12]
        )
        credit_history = st.selectbox(
            "Credit History", [1.0, 0.0],
            format_func=lambda value: (
                "Good (1)" if value == 1.0 else "Poor (0)"
            )
        )

    property_area = st.selectbox(
        "Property Area", ["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:

    # Match the feature names and order used during training
    applicant_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Preprocess the input using the fitted pipeline
    processed_data = preprocessor.transform(applicant_data)

    # Generate prediction and class probabilities
    prediction = model.predict(processed_data)[0]
    probabilities = model.predict_proba(processed_data)[0]

    # Find the probability corresponding to approved status
    approval_index = list(model.classes_).index("Y")
    approval_probability = probabilities[approval_index]

    st.subheader("Prediction Result")

    if prediction == "Y":
        st.success("Predicted Status: Approved")
    else:
        st.error("Predicted Status: Rejected")

    st.metric(
        "Estimated Approval Probability",
        f"{approval_probability:.2%}"
    )

    st.caption(
        "This probability is a model estimate, not a guarantee "
        "of approval or a real-world lending decision."
    )

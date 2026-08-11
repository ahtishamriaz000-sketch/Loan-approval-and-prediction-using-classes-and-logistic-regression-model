# ==========================================
# Loan Prediction System 
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# STREAMLIT PAGE SETTINGS
# ==========================================
st.set_page_config(page_title="Loan Prediction System", layout="wide")

st.title("🏦 Loan Prediction System from mian and mian saltanat")
st.write("Machine Learning se loan approve ya reject predict karna")


# ==========================================
# CLASS 1 : DATA CLEANING
# ==========================================
class DataCleaner:

    def clean_data(self, data):

        # Loan_ID remove
        data = data.drop("Loan_ID", axis=1)

        # Missing values fill
        for col in data.columns:

            if data[col].dtype == "int64" or data[col].dtype == "float64":
                data[col] = data[col].fillna(data[col].median())

            else:
                data[col] = data[col].fillna(data[col].mode()[0])

        return data


# ==========================================
# CLASS 2 : ENCODING
# ==========================================
class DataEncoder:

    def encode_data(self, data):

        data = pd.get_dummies(data, drop_first=True)

        return data


# ==========================================
# CLASS 3 : MODEL TRAINING
# ==========================================
class ModelTrainer:

    def train_model(self, X_train, y_train):

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        return model


# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])


if uploaded_file is not None:

    # ==========================================
    # LOAD DATA
    # ==========================================
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(data.head())

    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])


    # ==========================================
    # DATA CLEANING
    # ==========================================
    cleaner = DataCleaner()

    data_clean = cleaner.clean_data(data)


    # ==========================================
    # ENCODING
    # ==========================================
    encoder = DataEncoder()

    data_encoded = encoder.encode_data(data_clean)


    # ==========================================
    # SPLIT DATA
    # ==========================================
    X = data_encoded.drop("Loan_Status_Y", axis=1)

    y = data_encoded["Loan_Status_Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # ==========================================
    # MODEL TRAINING
    # ==========================================
    trainer = ModelTrainer()

    model = trainer.train_model(X_train, y_train)


    # ==========================================
    # MODEL TESTING
    # ==========================================
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.subheader("Model Accuracy")

    st.write(round(accuracy*100,2), "%")


    # ==========================================
    # CONFUSION MATRIX
    # ==========================================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    st.pyplot(fig)


    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================
    st.subheader("Classification Report")

    report = classification_report(y_test, predictions, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)


    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    st.subheader("Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots()

        sns.countplot(x="Loan_Status", data=data_clean)

        ax.set_title("Loan Status Distribution")

        st.pyplot(fig)


    with col2:

        fig, ax = plt.subplots()

        sns.histplot(data_clean["LoanAmount"], bins=20)

        ax.set_title("Loan Amount Distribution")

        st.pyplot(fig)


    # ==========================================
    # PREDICTION SECTION
    # ==========================================
    st.subheader("Predict New Loan")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Male","Female"])

        married = st.selectbox("Married", ["Yes","No"])

        dependents = st.selectbox("Dependents", ["0","1","2","3+"])

        education = st.selectbox("Education", ["Graduate","Not Graduate"])

        self_employed = st.selectbox("Self Employed", ["Yes","No"])

        credit_history = st.selectbox("Credit History", [1.0,0.0])


    with col2:

        applicant_income = st.number_input("Applicant Income",0)

        coapplicant_income = st.number_input("Coapplicant Income",0)

        loan_amount = st.number_input("Loan Amount",0)

        loan_term = st.selectbox(
            "Loan Term",
            [360,120,180,240,300,480,84,60,36,12]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban","Semiurban","Rural"]
        )


    if st.button("Predict Loan"):


        input_data = pd.DataFrame({

            "ApplicantIncome":[applicant_income],

            "CoapplicantIncome":[coapplicant_income],

            "LoanAmount":[loan_amount],

            "Loan_Amount_Term":[loan_term],

            "Credit_History":[credit_history],


            "Gender_Male":[1 if gender=="Male" else 0],

            "Married_Yes":[1 if married=="Yes" else 0],


            "Dependents_1":[1 if dependents=="1" else 0],

            "Dependents_2":[1 if dependents=="2" else 0],

            "Dependents_3+":[1 if dependents=="3+" else 0],


            "Education_Not Graduate":[1 if education=="Not Graduate" else 0],


            "Self_Employed_Yes":[1 if self_employed=="Yes" else 0],


            "Property_Area_Semiurban":[1 if property_area=="Semiurban" else 0],

            "Property_Area_Urban":[1 if property_area=="Urban" else 0]

        })


        # Missing columns add
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]


        result = model.predict(input_data)[0]


        if result == 1:

            st.success("Loan Approved")

        else:

            st.error("Loan Rejected")


else:

    st.info("Please upload dataset to start")
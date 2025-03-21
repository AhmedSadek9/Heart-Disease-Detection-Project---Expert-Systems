import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from experta import *
from rules import HeartDiseaseExpert
import io

# Load models and feature columns using relative paths
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_decision_tree_model.joblib')
features_path = os.path.join(os.path.dirname(__file__), 'model_features.joblib')
model = joblib.load(model_path)
model_features = joblib.load(features_path)

st.set_page_config(page_title="Heart Disease Detection System", layout="wide")

def run_expert_system(patient_data):
    st.write("\n🧠 Running Expert System Evaluation...")
    engine = HeartDiseaseExpert()
    engine.reset()
    for key, value in patient_data.items():
        engine.declare(Fact(**{key: value}))
    engine.run()

def run_decision_tree_model(patient_data):
    df_input = pd.DataFrame([patient_data])
    df_input_encoded = pd.get_dummies(df_input)

    for col in model_features:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    df_input_encoded = df_input_encoded[model_features]

    prediction = model.predict(df_input_encoded)[0]
    probability = model.predict_proba(df_input_encoded)[0][1] * 100

    if prediction == 1:
        st.error(f"Prediction: Likely to have heart disease (Confidence: {probability:.2f}%)")
    else:
        st.success(f"Prediction: Unlikely to have heart disease (Confidence: {100 - probability:.2f}%)")

    return prediction, probability

# Sidebar UI
def sidebar_input():
    st.sidebar.title("Input Health Information")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
    cholesterol = st.sidebar.number_input("Cholesterol Level", min_value=0)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0)
    smoking = st.sidebar.selectbox("Do you smoke?", ["yes", "no"])
    exercise = st.sidebar.selectbox("Exercise frequency", ["none", "irregular", "regular"])
    bmi = st.sidebar.number_input("BMI", min_value=0.0, format="%.2f")
    glucose = st.sidebar.number_input("Glucose Level", min_value=0)
    family_history = st.sidebar.selectbox("Family history of heart disease?", ["yes", "no"])
    diet = st.sidebar.selectbox("Diet type", ["healthy", "unhealthy"])
    alcohol = st.sidebar.selectbox("Do you consume alcohol?", ["yes", "no"])
    cp = st.sidebar.number_input("Chest pain type (1/2/3/4)", min_value=1, max_value=4)
    restecg = st.sidebar.number_input("RestECG (0/1/2)", min_value=0, max_value=2)
    exang = st.sidebar.number_input("Exercise induced angina (0/1)", min_value=0, max_value=1)
    oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, format="%.2f")
    slope = st.sidebar.number_input("Slope (1/2/3)", min_value=1, max_value=3)
    ca = st.sidebar.number_input("CA (0/1/2/3/4)", min_value=0, max_value=4)
    thal = st.sidebar.number_input("Thal (0/1/2/3)", min_value=0, max_value=3)
    fbs = st.sidebar.number_input("Fasting blood sugar >120 (0/1)", min_value=0, max_value=1)

    return {
        "age": age,
        "cholesterol": cholesterol,
        "blood_pressure": blood_pressure,
        "smoking": smoking,
        "exercise": exercise,
        "bmi": bmi,
        "glucose": glucose,
        "family_history": family_history,
        "diet": diet,
        "alcohol": alcohol,
        "cp": cp,
        "restecg": restecg,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "fbs": fbs
    }

# Health Tips Based on User Input
def show_health_tips(prediction):
    st.subheader("💡 Health Recommendations")
    if prediction == 1:
        st.write("- Consult a cardiologist.")
        st.write("- Maintain a healthy diet and exercise regularly.")
        st.write("- Avoid smoking and alcohol consumption.")
        st.write("- Monitor cholesterol, blood pressure, and glucose levels.")
    else:
        st.write("- Keep up with healthy habits!")
        st.write("- Regular exercise and balanced diet are key.")

# Download Report
def download_report(data, prediction, probability):
    report = pd.DataFrame.from_dict(data, orient='index', columns=['Values'])
    report.loc['Prediction'] = 'Likely' if prediction == 1 else 'Unlikely'
    report.loc['Confidence'] = f"{probability:.2f}%"

    buffer = io.StringIO()
    report.to_csv(buffer)
    st.download_button("Download Prediction Report", data=buffer.getvalue(), file_name="heart_disease_report.csv")

# Main app
st.title("❤️ Heart Disease Detection System")

user_data = sidebar_input()

if st.sidebar.button("Predict"):
    st.subheader("Expert System Results:")
    run_expert_system(user_data)

    st.subheader("Decision Tree Model Prediction:")
    prediction, probability = run_decision_tree_model(user_data)

    show_health_tips(prediction)

    st.subheader("📊 Visualization Dashboard")

    # Visualizing key metrics
    sample_data = pd.DataFrame({
        'Metric': ['Cholesterol', 'Blood Pressure', 'BMI', 'Glucose'],
        'Value': [user_data['cholesterol'], user_data['blood_pressure'], user_data['bmi'], user_data['glucose']]
    })

    fig, ax = plt.subplots()
    sns.barplot(x='Metric', y='Value', data=sample_data, palette='Reds')
    st.pyplot(fig)

    # Pie chart (mock-up contribution visualization)
    st.subheader("Risk Factor Contribution (Illustrative)")
    pie_data = pd.Series({
        'Cholesterol': 30,
        'Blood Pressure': 25,
        'BMI': 20,
        'Glucose': 25
    })
    fig2, ax2 = plt.subplots()
    ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    st.pyplot(fig2)

    download_report(user_data, prediction, probability)

st.sidebar.markdown("---")
st.sidebar.write("Developed for academic purposes. Ready for cloud deployment.")

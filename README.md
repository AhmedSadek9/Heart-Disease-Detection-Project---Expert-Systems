# ❤️Heart Disease Detection System❤️

## 📋 Project Overview
This project is a comprehensive Heart Disease Detection System that combines a rule-based expert system and a machine learning model (Decision Tree Classifier). It provides both medical reasoning and predictive analytics based on patient input, with a user-friendly web interface built using Streamlit.

## ✅ Key Features
- **Rule-Based Expert System** powered by the Experta library.
- **Decision Tree Classifier** trained with hyperparameter tuning.
- **Streamlit web application** for easy and interactive user input.
- Dynamic **confidence-based predictions** blending expert and model assessments.
- Risk factor visualizations (bar chart and pie chart) based on personal data.
- Downloadable patient report in CSV format.
- Extensive input parameters: cholesterol, blood pressure, BMI, glucose, exercise, smoking, alcohol, stress, sleep hours, family history, chest pain, and more.

## 📁 Project Structure
```
│── rule_based_system/
│   ├── rules.py                      # Expert system rules
│   ├── expert_system.py              # Command-line expert system execution
│   ├── Decision Tree Model.py        # Model training and evaluation script
│
│── model_features.joblib             # Stored feature columns post encoding
│── heart_disease_decision_tree_model.joblib   # Trained decision tree model
│── heart_cleaned_data.csv            # Dataset used for training
│── main.py                           # Streamlit-based main interface
│── README.md                         # Documentation and setup guide
```

## ⚙️ Setup Instructions
1. **Clone the repository**:
```bash
git clone <repository_url>
```
2. **Navigate to the project directory**:
```bash
cd heart-disease-detection-system
```
3. **📦 Key Package Requirements**:
 ```bash
numpy>=1.26.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
experta==1.9.2   # experta==1.9.4 (optional update)
graphviz>=0.20.1
frozendict==2.3.8
scipy==1.7.1
```
 4. **Install dependencies**:
```bash
pip install -r requirements.txt
```
5. **Run the Streamlit App**:
```bash
streamlit run "D:\Semester Four\Intelligent Programming\Project 1\notebooks\rule_based_system\app.py"   ##Change path of file (Run on CMD)
```

## 🌐 Streamlit Application Details
- The app provides a sidebar with comprehensive health input fields (age, cholesterol, BP, glucose, BMI, etc.).
- After input submission, it shows expert system outputs and model-based predictions with confidence levels.
- Interactive visualizations:
  - Bar charts for key health metrics.
  - Pie charts showing factor contributions to overall risk.
- Downloadable reports for users in CSV format.
- Responsive UI with wide layout support for a smooth experience.
- Ready to be deployed on Streamlit Cloud or Heroku for remote access.

## 📊 Dataset Details
The dataset includes features such as:
- Age, cholesterol level, blood pressure, glucose, BMI
- Exercise frequency, smoking status, alcohol consumption
- Diet type, family history, chest pain type, rest ECG results, and more.
- Source: [Add dataset source or citation if applicable]

## 🧠 Expert System Rules Summary
- High cholesterol combined with age triggers warning.
- Blood pressure issues and smoking raise alerts.
- Poor diet and obesity contribute to risks.
- Genetic factors combined with unhealthy lifestyle conditions are highlighted.
- Inactivity and stress levels are factored into risk.
- Real-time explanations of triggered rules are shown to the user.

## 🤖 Machine Learning Model
- Decision Tree Classifier trained with GridSearchCV.
- Key metrics: Accuracy, Precision, Recall, F1-Score.
- Stored using `joblib` for fast inference.
- Combined prediction confidence calculated using probability and custom risk score.

## 📈 Key Visualizations in the App
- Bar plot: Cholesterol, Blood Pressure, BMI, Glucose, Sleep Hours.
- Pie chart: Relative contribution of cholesterol, blood pressure, BMI, glucose, and sleep & stress factors.
- Downloadable CSV report of user inputs and prediction.




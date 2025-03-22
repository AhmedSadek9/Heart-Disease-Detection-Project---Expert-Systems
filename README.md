# Heart Disease Detection System

## 📋 Project Overview
This project is a comprehensive Heart Disease Detection System combining a rule-based expert system and a machine learning model (Decision Tree Classifier). The system provides expert reasoning and predictive analytics based on patient input.

## ✅ Key Features
- **Rule-Based Expert System** built with the Experta library.
- **Decision Tree Classifier** trained with hyperparameter tuning.
- **User input validation** with clear prompts and safe ranges.
- **Confidence-based prediction** with accuracy calculation.
- **Feature importance visualization (planned)**.
- Support for additional parameters like exercise habits, smoking status, glucose levels, family history, and more.

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
│── main.py                           # Combined expert system + model CLI script
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
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Details
The dataset includes features such as:
- Age, cholesterol level, blood pressure, glucose, BMI
- Exercise frequency, smoking status, alcohol consumption
- Diet type, family history, chest pain type, and more.

## 🧠 Expert System Rules Summary
- High cholesterol combined with age
- High blood pressure with smoking
- Obesity with poor diet
- Genetic risks with age
- Inactivity and high BMI
- Alcohol and blood pressure risks
- All conditions print real-time risk warnings.

## 🤖 Machine Learning Model
- Trained Decision Tree Classifier with GridSearchCV tuning.
- Performance metrics: Accuracy, Precision, Recall, and F1-Score.
- Model saved via `joblib` for deployment.

## 🚀 How to Run
1. **Run both Expert System & Model Prediction:**
```bash
python main.py
```
2. **Train the Decision Tree Model (if retraining is required):**
```bash
python rule_based_system/Decision Tree Model.py
```

## 📈 Planned Enhancements
- Interactive Streamlit web app interface.
- Dynamic visualization of user-based risk contributions.
- Detailed risk factor explanations and education.
- Deployment as a web service.
- Integration of additional algorithms for ensemble prediction.




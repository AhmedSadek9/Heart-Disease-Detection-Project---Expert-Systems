 #File: rule_based_system/expert_system.py
from experta import *
from rules import HeartDiseaseExpert

def get_int_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))     
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_float_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_choice_input(prompt, choices):
    while True:
        value = input(prompt).lower()
        if value in choices:
            return value
        else:
            print(f"Invalid choice. Please enter one of {choices}.")

def get_user_input():
    patient_data = {
        "age": get_int_input("Enter age: ", min_value=0, max_value=120),
        "cholesterol": get_int_input("Enter cholesterol level: ", min_value=0),
        "blood_pressure": get_int_input("Enter blood pressure: ", min_value=0),
        "smoking": get_choice_input("Do you smoke? (yes/no): ", ["yes", "no"]),
        "exercise": get_choice_input("Exercise frequency (none/irregular/regular): ", ["none", "irregular", "regular"]),
        "bmi": get_float_input("Enter BMI: ", min_value=0),
        "glucose": get_int_input("Enter glucose level: ", min_value=0),
        "family_history": get_choice_input("Family history of heart disease? (yes/no): ", ["yes", "no"]),
        "diet": get_choice_input("Diet type (healthy/unhealthy): ", ["healthy", "unhealthy"]),
        "alcohol": get_choice_input("Do you consume alcohol? (yes/no): ", ["yes", "no"])
    }
    return patient_data

def run_expert_system(patient_data):
    engine = HeartDiseaseExpert()
    engine.reset()
    for key, value in patient_data.items():
        engine.declare(Fact(**{key: value}))
    engine.run()

if __name__ == "__main__":
    patient_data = get_user_input()
    run_expert_system(patient_data)
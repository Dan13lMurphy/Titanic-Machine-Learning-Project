# titanic_web_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load / Train the model
# -----------------------

# Load training data
train = pd.read_csv("train.csv")

# Data cleaning
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns=['Embarked'])

# Features & labels
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Ensure Embarked columns exist
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in train:
        train[col] = 0

X = train[features]
y = train['Survived']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------
# Streamlit web interface
# -----------------------

st.title("Would You Survive the Titanic? 🛳️")
st.write("""
This app predicts your survival probability on the Titanic based on historical data.
You can adjust your details below to see your likelihood of survival!
""")

# -----------------------
# Titanic overall statistics
# -----------------------
st.subheader("Titanic Survival Statistics 📊")
total_passengers = len(train)
overall_survival_rate = train['Survived'].mean() * 100
male_survival = train[train['Sex'] == 0]['Survived'].mean() * 100
female_survival = train[train['Sex'] == 1]['Survived'].mean() * 100

st.markdown(f"- Total passengers in dataset: **{total_passengers}**")
st.markdown(f"- Overall survival rate: **{overall_survival_rate:.1f}%**")
st.markdown(f"- Male survival rate: **{male_survival:.1f}%**")
st.markdown(f"- Female survival rate: **{female_survival:.1f}%**")

# -----------------------
# User inputs with explanations
# -----------------------

st.subheader("Enter your details:")

sex = st.selectbox("Sex", ["Male", "Female"],
                   help="Biological sex. Historically, women and children had higher survival rates.")
pclass = st.selectbox("Passenger Class", [
                      1, 2, 3], help="Ticket class: 1 = Upper, 2 = Middle, 3 = Lower. Higher classes had better survival chances.")
age = st.slider("Age", 0, 100, 25,
                help="Age of passenger. Children had higher survival chances.")
sibsp = st.slider("Number of siblings/spouses aboard", 0, 10, 0,
                  help="Number of siblings/spouses aboard. Family size could affect survival.")
parch = st.slider("Number of parents/children aboard", 0, 10, 0,
                  help="Number of parents/children aboard. Family presence affected survival.")
fare = st.number_input("Ticket Fare ($)", min_value=0.0, value=32.0,
                       help="Price of the ticket. Higher fare (higher class) could indicate better survival odds.")
embarked = st.selectbox("Port of Embarkation", [
                        "C", "Q", "S"], help="Where the passenger boarded: C = Cherbourg, Q = Queenstown, S = Southampton")

# Convert inputs to numeric for model
sex_val = 0 if sex == "Male" else 1
embarked_vals = [0, 0, 0]
if embarked == "C":
    embarked_vals[0] = 1
if embarked == "Q":
    embarked_vals[1] = 1
if embarked == "S":
    embarked_vals[2] = 1

user_data = np.array([[pclass, sex_val, age, sibsp, parch, fare,
                       embarked_vals[0], embarked_vals[1], embarked_vals[2]]])

# Predict button
if st.button("Check Survival"):
    prediction = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0][1]  # probability of survival

    st.subheader("Prediction Result 📝")
    if prediction == 1:
        st.success(f"You would likely survive! (Probability: {prob:.2f})")
    else:
        st.error(f"You would likely not survive. (Probability: {prob:.2f})")

    st.info("This prediction is based on historical Titanic data and statistical probabilities, not a guarantee.")

# Optional: show feature influence
st.subheader("How features affected survival historically:")
st.markdown("""
- **Sex:** Women had much higher survival rates than men.  
- **Passenger Class:** 1st class passengers had the highest survival, 3rd class the lowest.  
- **Age:** Children had higher survival odds.  
- **Family Aboard:** Small families often survived better; very large families struggled.  
- **Fare:** Higher fare often meant better conditions and higher survival.
""")

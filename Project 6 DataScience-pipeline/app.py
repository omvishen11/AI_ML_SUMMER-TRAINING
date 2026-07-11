import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Page Setup
st.title("🏃‍♂️ Employee Retention Predictor")
st.write("Adjust the sliders below to predict if an employee will stay or leave the company.")

# 2. Cache the data loading and model training so it doesn't re-run on every click
@st.cache_data
def train_model():
    # Load data
    df = pd.read_csv('HR_comma_sep.csv')
    
    # Isolate variables
    subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
    
    # Create dummy variables for salary
    salary_dummies = pd.get_dummies(subdf['salary'], prefix="salary")
    df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
    df_with_dummies.drop('salary', axis='columns', inplace=True)
    
    X = df_with_dummies
    y = df['left']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy, X.columns

# Load the model and columns
try:
    model, accuracy, feature_cols = train_model()
    st.success(f"Model trained successfully! Accuracy: **{accuracy * 100:.2f}%**")
except FileNotFoundError:
    st.error("Error: 'HR_comma_sep.csv' not found. Please upload it to your directory.")
    st.stop()

st.divider()

# 3. Create the User Interface for predictions
st.subheader("Predict an Employee's Future")

col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    hours = st.slider("Average Monthly Hours", 50, 350, 200)

with col2:
    promotion = st.radio("Promoted in the last 5 years?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# 4. Process the user input to match the model's expected format
input_data = {
    'satisfaction_level': satisfaction,
    'average_montly_hours': hours,
    'promotion_last_5years': promotion,
    'salary_high': 1 if salary == 'high' else 0,
    'salary_low': 1 if salary == 'low' else 0,
    'salary_medium': 1 if salary == 'medium' else 0
}

# Convert dictionary to DataFrame and ensure columns are in the exact same order as training
input_df = pd.DataFrame([input_data])[feature_cols]

# 5. Make the Prediction
if st.button("Predict 🔮", type="primary"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Risk! This employee is likely to LEAVE. (Probability: {probability*100:.1f}%)")
    else:
        st.success(f"🎉 Safe! This employee is likely to STAY. (Probability: {(1-probability)*100:.1f}%)")

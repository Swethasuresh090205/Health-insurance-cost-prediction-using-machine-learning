import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sidebar navigation
nav = st.sidebar.radio("Navigation", ["About", "Predict"])

# Load the dataset
try:
    df = pd.read_csv(r'C:\Users\sures\project\env\insurance.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

# Data preprocessing
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Features and target variable
x = df.drop(columns='charges', axis=1)
y = df['charges']

# Train the model globally
rfr = RandomForestRegressor()
rfr.fit(x, y)

# Model performance metrics
y_pred = rfr.predict(x)


if nav == "About":
    st.title("Health Insurance Premium Predictor")
    st.text("This application predicts health insurance premiums based on user input.")
    
    # Add an image (make sure to provide the correct path or URL)
    st.image(r'C:\Users\sures\project\compare-health-insurance-plans.jpg', caption='Understanding Health Insurance', use_container_width=True)

    # About section content...

if nav == "Predict":
    st.title("Enter Details")

    # User input fields with validation
    age = st.number_input("Age: ", step=1, min_value=0)
    
    sex = st.radio("Sex", ("Male", "Female"))
    s = 0 if sex == "Male" else 1
    
    bmi = st.number_input("BMI (Body Mass Index): ", min_value=10.0, max_value=50.0)  # Set reasonable limits
    
    children = st.number_input("Number of children: ", step=1, min_value=0)
    
    smoke = st.radio("Do you smoke?", ("Yes", "No"))
    sm = 0 if smoke == "Yes" else 1
    
    region = st.selectbox('Region', ('SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'))
    reg = {'SouthEast': 0, 'SouthWest': 1, 'NorthEast': 2, 'NorthWest': 3}[region]

    # Prediction button
    if st.button("Predict"):
        predicted_premium = rfr.predict([[age, s, bmi, children, sm, reg]])
        
        # Displaying predicted premium in INR
        st.subheader("Predicted Premium")
        st.text(f"₹{predicted_premium[0]:,.2f}")  

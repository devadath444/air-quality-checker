import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.title("Air Quality Checker")

# Load dataset directly
@st.cache_data
def load_data():
    data = pd.read_csv('AirQuality.csv', sep=';', decimal=',')
    data = data.iloc[:, :-2]  # Drop last 2 empty columns
    data = data.dropna()
    data = data.apply(pd.to_numeric, errors='coerce')
    if 'Date' in data.columns and 'Time' in data.columns:
        data = data.drop(columns=["Date", "Time"])
    return data

data = load_data()

st.subheader("Cleaned Dataset Preview")
st.write(data.head())

st.subheader("Dataset Info")
st.text(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

if 'NO2(GT)' not in data.columns:
    st.error("Required column 'NO2(GT)' not found in dataset.")
else:
    x = data.drop("NO2(GT)", axis=1)
    y = data['NO2(GT)']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.transform(x_test)

    model_type = st.selectbox("Select a model", ["Decision Tree", "Support Vector Machine"])

    # Train the selected model
    if model_type == "Decision Tree":
        model = DecisionTreeRegressor()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scaler_used = False
    else:
        model = SVR()
        model.fit(x_train_scale, y_train)
        predictions = model.predict(x_test_scale)
        scaler_used = True

    # Show accuracy metrics
    st.subheader("Model Evaluation on Test Set")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared Score (R²): {r2:.2f}")

    # Create sliders for user input based on features
    st.subheader("Enter Feature Values")
    user_input = []
    for col in x.columns:
        min_val = float(x[col].min())
        max_val = float(x[col].max())
        mean_val = float(x[col].mean())
        val = st.slider(f"{col}", min_val, max_val, mean_val)
        user_input.append(val)

    user_array = np.array(user_input).reshape(1, -1)
    if scaler_used:
        user_array = scaler.transform(user_array)

    if st.button("Predict Air Quality"):
        predicted_no2 = model.predict(user_array)[0]

        def classify_air_quality(value):
            if value <= 40:
                return "Good"
            elif value <= 80:
                return "Moderate"
            elif value <= 180:
                return "Unhealthy"
            else:
                return "Very Unhealthy"

        air_quality_label = classify_air_quality(predicted_no2)
        st.success(f"Predicted NO2 Level: {predicted_no2:.2f} → Air Quality: {air_quality_label}")

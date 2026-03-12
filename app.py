import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("💳 Fraud Detection System")
st.write("This app detects fraudulent transactions using Machine Learning")

# Generate fake data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'amount': np.random.exponential(100, n),
    'time': np.random.randint(0, 24, n),
    'distance_from_home': np.random.exponential(50, n),
    'is_fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])
})

# Train model
X = data[['amount', 'time', 'distance_from_home']]
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Show accuracy
st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# User Input
st.subheader("🔍 Check a Transaction")
amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=100.0)
time = st.slider("Hour of Transaction", 0, 23, 12)
distance = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0)

if st.button("Check Fraud"):
    input_data = pd.DataFrame([[amount, time, distance]],
                              columns=['amount', 'time', 'distance_from_home'])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("✅ Transaction looks SAFE!")
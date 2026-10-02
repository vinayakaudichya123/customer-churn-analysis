import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📊",
    layout="wide"
)

# Load model
preprocessor, rf_model = joblib.load("sales_model.pkl")


# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 30px;
}

.section {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# Title
st.markdown(
    '<div class="title">📊 Sales Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict expected sales using Machine Learning</div>',
    unsafe_allow_html=True
)


# Customer & Order Details
st.markdown(
    '<div class="section">👤 Customer & Order Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

with col2:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=100,
        value=2
    )

with col3:
    unit_price = st.number_input(
        "Unit Price",
        min_value=0.0,
        value=1000.0
    )

with col4:
    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )


# Product Details
st.markdown(
    '<div class="section">🛍️ Product Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    category = st.selectbox(
        "Category",
        ["Electronics", "Clothing", "Groceries"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        ["Cash", "Card", "UPI"]
    )

with col3:
    city = st.text_input(
        "City",
        "Jaipur"
    )

with col4:
    state = st.text_input(
        "State",
        "Rajasthan"
    )


# Date Details
st.markdown(
    '<div class="section">📅 Date Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2030,
        value=2026
    )

with col2:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=9
    )

with col3:
    day_of_week = st.number_input(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )

with col4:
    is_weekend = st.selectbox(
        "Is Weekend?",
        [0, 1]
    )


# Prediction Button
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Sales", use_container_width=True):

    new_data = pd.DataFrame({
        "Age": [age],
        "Quantity": [quantity],
        "Unit_Price": [unit_price],
        "Discount": [discount],
        "Category": [category],
        "Payment_Method": [payment_method],
        "City": [city],
        "State": [state],
        "Year": [year],
        "Month": [month],
        "DayOfWeek": [day_of_week],
        "IsWeekend": [is_weekend]
    })

    # Preprocess input
    new_data_encoded = preprocessor.transform(new_data)

    # Prediction
    prediction = rf_model.predict(new_data_encoded)

    # Result
    st.markdown(
        f"""
        <div class="result">
            <h2>💰 Predicted Sales</h2>
            <h1>₹ {prediction[0]:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
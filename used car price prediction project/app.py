import streamlit as st
from main import predict_price
from datetime import datetime

st.title("Used Car Price Prediction App")

# User inputs
company = st.text_input("Company")
model = st.text_input("Model")
variant = st.text_input("Variant")
fuel_type = st.selectbox("Fuel Type", ['PETROL', 'DIESEL', 'CNG', 'LPG', 'HYBRID'])
colour = st.text_input("Colour")
kilometer = st.number_input("Kilometers Driven", 0)
body_style = st.selectbox("Body Style", ['HATCHBACK', 'SEDAN', 'SUV', 'MUV','MPV','COMPACTSUV','VAN'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
manufacture_date = st.date_input("Manufacture Date")
model_year = st.number_input("Model Year", min_value=2000, max_value=2023)
cngkit = st.selectbox("CNG Kit", ['Yes', 'No'])
owner = st.selectbox("Owner Type", ['1st Owner', '2nd Owner', '3rd Owner', '4th Owner'])
dealer_state = st.text_input("Dealer State")
dealer_name = st.text_input("Dealer Name")
city = st.text_input("City")
warranty = st.selectbox("Warranty Available?", ['Yes', 'No'])
quality_score = st.slider("Quality Score", 0.0, 10.0, 5.0)

current_year = datetime.now().year
car_age = current_year - model_year

input_dict = {
    "Company": company,
    "Model": model,
    "Variant": variant,
    "FuelType": fuel_type,
    "Colour": colour,
    "Kilometer": kilometer,
    "BodyStyle": body_style,
    "TransmissionType": transmission,
    "ModelYear": model_year,
    "Owner": owner,
    "DealerState": dealer_state,
    "DealerName": dealer_name,
    "City": city,
    "Warranty": warranty,
    "QualityScore": quality_score,
    "Car_Age": car_age   # ✅ FIXED NAME
}

if st.button("Predict Price"):
    prediction = predict_price(input_dict)
    st.success(f"Predicted Price: ₹ {round(prediction, 2)} Lakhs")


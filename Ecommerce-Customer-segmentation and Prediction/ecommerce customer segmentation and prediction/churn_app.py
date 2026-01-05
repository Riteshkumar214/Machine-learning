import streamlit as st
import pandas as pd
import joblib   
from churn_main import load_data, create_rfm, preprocess_rfm, train_kmeans
segment_map = {
    0: "VIP Customer",
    1: "Loyal Customer",
    2: "Low Value Customer",
    3: "Inactive Customer"
}


st.title("📊 E-Commerce Customer Segmentation & Prediction App")

menu = st.sidebar.selectbox(
    "Select Option",
    ["Upload Data", "Train Model", "Predict Segment"]
)


# ---------------------- UPLOAD DATA -------------------------
if menu == "Upload Data":
    st.header("📥 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding="latin-1")
        st.dataframe(df.head())
        df.to_csv("data.csv", index=False)
        st.success("File uploaded and saved as data.csv")


# ---------------------- TRAIN MODEL -------------------------
elif menu == "Train Model":
    st.header("🧠 Train Model")

    try:
        df = load_data("data.csv")
        rfm = create_rfm(df)
        X_scaled, scaler = preprocess_rfm(rfm)
        kmeans, clusters = train_kmeans(X_scaled, n_clusters=4)

        rfm["Cluster"] = clusters

        st.subheader("Cluster Distribution")
        st.bar_chart(rfm["Cluster"].value_counts())

        # ---------------- SAVE MODELS ----------------
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(kmeans, "kmeans_model.pkl")

        st.success("Model trained successfully! Now you can predict.")

        st.session_state["scaler"] = scaler
        st.session_state["model"] = kmeans

    except Exception as e:
        st.error(f"Error: {e}")


# ---------------------- PREDICTION -------------------------
elif menu == "Predict Segment":
    st.header("🔮 Predict Customer Segment")

    if "model" not in st.session_state:
        st.warning("Train the model first!")

    else:
        scaler = st.session_state["scaler"]
        model = st.session_state["model"]

        recency = st.number_input("Recency (days)", min_value=0)
        frequency = st.number_input("Frequency", min_value=0)
        monetary = st.number_input("Total Price", min_value=0)

        if st.button("Predict"):
            X_new = [[recency, frequency, monetary]]
            X_scaled = scaler.transform(X_new)

            segment = model.predict(X_scaled)[0]
            customer_type = segment_map.get(segment, "Unknown")

            st.success(f"Customer Type: {customer_type}")
            st.info(f"(Internal Segment ID: {segment})")

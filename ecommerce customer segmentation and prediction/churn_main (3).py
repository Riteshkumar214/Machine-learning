import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.drop_duplicates()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def create_rfm(df):
    rfm_m = df.groupby("CustomerID")["TotalPrice"].sum().reset_index()
    rfm_f = df.groupby("CustomerID")["InvoiceNo"].count().reset_index()
    rfm_f.columns = ["CustomerID", "Frequency"]

    df["Recency"] = (df["InvoiceDate"].max() - df["InvoiceDate"]).dt.days
    rfm_r = df.groupby("CustomerID")["Recency"].min().reset_index()

    rfm = rfm_m.merge(rfm_f, on="CustomerID").merge(rfm_r, on="CustomerID")
    return rfm


def preprocess_rfm(rfm):
    # Outlier capping
    for col in ["Recency", "Frequency", "TotalPrice"]:
        rfm[col] = np.where(
            rfm[col] > rfm[col].quantile(0.95),
            rfm[col].quantile(0.95),
            rfm[col]
        )

    X = rfm[["Recency", "Frequency", "TotalPrice"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def train_kmeans(X_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Saving only model here
    joblib.dump(kmeans, "kmeans_model.pkl")

    return kmeans, clusters

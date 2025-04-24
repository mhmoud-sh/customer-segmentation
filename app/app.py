import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from utils.preprocessing import load_data, preprocess_data
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation App")

# Load data
df = load_data("data/ecommerce_data.csv")
st.subheader("Raw Data Preview")
st.write(df.head())

# Select features
features = st.multiselect("Select Features for Clustering:", df.columns.tolist(), default=df.columns[:4].tolist())

if features:
    # Preprocess and cluster
    X_scaled, scaler = preprocess_data(df, features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    df["Cluster"] = labels
    st.success("✅ Customers clustered successfully!")

    # Plot
    if len(features) >= 2:
        fig = px.scatter(df, x=features[0], y=features[1], color="Cluster", title="Customer Segments")
        st.plotly_chart(fig, use_container_width=True)

    # Download cluster results
    st.download_button("📥 Download Clustered Data", df.to_csv(index=False), "clustered_customers.csv")

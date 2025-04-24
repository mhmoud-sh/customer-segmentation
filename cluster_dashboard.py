import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# Load your clustered data
rfm_df = pd.read_csv("your_clustered_data.csv")  # Replace with your actual file

# Set page config
st.set_page_config(layout="wide", page_title="RFM Cluster Dashboard")

st.title("🧠 RFM Customer Segmentation Visualization")

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA Clusters", "t-SNE Clusters"))

pca_fig = px.scatter(
    rfm_df, x='PCA1', y='PCA2', color='Cluster',
    hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary']
)

tsne_fig = px.scatter(
    rfm_df, x='TSNE1', y='TSNE2', color='Cluster',
    hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary']
)

for trace in pca_fig.data:
    fig.add_trace(trace, row=1, col=1)

for trace in tsne_fig.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(height=600, showlegend=False)

st.plotly_chart(fig, use_container_width=True)

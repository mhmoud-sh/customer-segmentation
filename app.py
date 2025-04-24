import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import uuid

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {background-color: #f9fafb;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 8px;}
    .stTabs {background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stFileUploader {background-color: #ffffff; padding: 10px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

# Preprocess Data
def preprocess_data(df):
    df = df[df['Quantity'] > 0]
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    df = df.dropna(subset=["CustomerID"])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df

# RFM Feature Engineering
def create_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# Clustering Function
def perform_kmeans(rfm, k=4):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Ensure numeric types
    rfm_kmeans = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_kmeans['Recency'] = pd.to_numeric(rfm_kmeans['Recency'], errors='coerce')
    rfm_kmeans['Frequency'] = pd.to_numeric(rfm_kmeans['Frequency'], errors='coerce')
    rfm_kmeans['Monetary'] = pd.to_numeric(rfm_kmeans['Monetary'], errors='coerce')

    # Handle any NaN values
    rfm_kmeans = rfm_kmeans.dropna()

    # Scale
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_kmeans)

    # Fit KMeans
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(rfm_scaled)

    # Add cluster labels
    rfm.loc[rfm_kmeans.index, 'Cluster'] = clusters
    return rfm

# Download Helper
def download_link(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return st.download_button(
        label="📥 Download Clustered Data as CSV",
        data=buffer.getvalue(),
        file_name="rfm_clusters.csv",
        mime="text/csv",
        key=str(uuid.uuid4())
    )

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("📂 Upload OnlineRetail.xlsx", type=["xlsx"])
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This dashboard segments customers based on RFM (Recency, Frequency, Monetary) metrics using KMeans clustering.")

# Main App
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("Analyze customer behavior and tailor marketing strategies with RFM-based clustering.")

if uploaded:
    with st.container():
        df = load_data(uploaded)
        df = preprocess_data(df)
        rfm = create_rfm(df)
        rfm = perform_kmeans(rfm, k=num_clusters)

        # Summary Statistics
        st.subheader("📊 Dataset Summary")
        summary = rfm.describe().round(2)
        st.dataframe(summary, use_container_width=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📋 RFM Table", "📊 Cluster Metrics", "📈 Interactive Visuals"])

        with tab1:
            st.subheader("📋 RFM Table with Cluster Labels")
            st.dataframe(rfm.sort_values(by="Cluster"), height=300, use_container_width=True)
            download_link(rfm)

        with tab2:
            st.subheader("📊 Cluster Profile Comparison")
            cluster_summary = rfm.groupby("Cluster").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "CustomerID": "count"
            }).round(1).reset_index()
            cluster_summary.rename(columns={"CustomerID": "Customer Count"}, inplace=True)

            # Bar Plot
            melted_summary = cluster_summary.melt(id_vars="Cluster", value_vars=["Recency", "Frequency", "Monetary"],
                                                 var_name="Metric", value_name="Mean Value")
            bar_fig = px.bar(
                melted_summary,
                x="Metric",
                y="Mean Value",
                color="Cluster",
                barmode="group",
                text="Mean Value",
                title="📊 Average RFM Scores per Cluster",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            bar_fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:.1f}")
            bar_fig.update_layout(
                yaxis_title="Average Value",
                xaxis_title="RFM Metric",
                hovermode="closest",
                showlegend=True
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Cluster Summary Table
            st.subheader("📋 Cluster Summary")
            st.dataframe(cluster_summary, use_container_width=True)

        with tab3:
            st.subheader("📈 RFM Cluster Distribution")
            scatter_fig = px.scatter(
                rfm,
                x="Recency",
                y="Monetary",
                color="Cluster",
                size="Frequency",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                title="🧭 RFM Cluster Distribution (Recency vs Monetary)",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            scatter_fig.update_traces(
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="CustomerID: %{customdata[0]}<br>Recency: %{x}<br>Monetary: %{y}<br>Frequency: %{customdata[2]}"
            )
            scatter_fig.update_layout(
                xaxis_title="Recency (days)",
                yaxis_title="Monetary Value (£)",
                hovermode="closest",
                showlegend=True
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Business Insights
        with st.expander("💡 Tailored Marketing Strategies by Cluster", expanded=True):
            st.markdown("""
            Based on the RFM clusters, here are tailored marketing strategies:
            - **Champions (Low Recency, High Frequency, High Monetary)**: Offer exclusive rewards, VIP programs, or early access to new products.
            - **Loyal Customers (Low Recency, Medium-High Frequency)**: Encourage referrals with incentives, provide loyalty discounts.
            - **Potential Loyalists (Moderate Recency, Medium Frequency)**: Send personalized follow-ups, recommend complementary products.
            - **At Risk (High Recency, Low Frequency)**: Launch win-back campaigns, offer discounts, or conduct surveys to understand churn.
            """)

else:
    st.warning("📁 Please upload the `OnlineRetail.xlsx` dataset to begin.")
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    .stButton>button {background-color: #1e40af; color: white; border-radius: 8px; padding: 8px 16px;}
    .stTabs {background-color: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .stFileUploader {background-color: #ffffff; padding: 12px; border-radius: 8px;}
    h1, h2, h3 {color: #1e40af; font-family: 'Arial', sans-serif;}
    .st-expander {border: 1px solid #e2e8f0; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Preprocess Data
@st.cache_data
def preprocess_data(df):
    progress = st.progress(0)
    df = df[df['Quantity'] > 0]
    progress.progress(25)
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    progress.progress(50)
    df = df.dropna(subset=["CustomerID"])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    progress.progress(100)
    return df

# RFM Feature Engineering
@st.cache_data
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
@st.cache_data
def perform_kmeans(rfm, k=4):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    rfm_kmeans = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_kmeans['Recency'] = pd.to_numeric(rfm_kmeans['Recency'], errors='coerce')
    rfm_kmeans['Frequency'] = pd.to_numeric(rfm_kmeans['Frequency'], errors='coerce')
    rfm_kmeans['Monetary'] = pd.to_numeric(rfm_kmeans['Monetary'], errors='coerce')
    rfm_kmeans = rfm_kmeans.dropna()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_kmeans)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(rfm_scaled)

    rfm.loc[rfm_kmeans.index, 'Cluster'] = clusters
    return rfm

# Cluster Naming
def name_clusters(rfm):
    cluster_names = {}
    for cluster in rfm['Cluster'].unique():
        cluster_data = rfm[rfm['Cluster'] == cluster]
        recency = cluster_data['Recency'].mean()
        frequency = cluster_data['Frequency'].mean()
        monetary = cluster_data['Monetary'].mean()
        if recency < rfm['Recency'].quantile(0.25) and frequency > rfm['Frequency'].quantile(0.75) and monetary > rfm['Monetary'].quantile(0.75):
            cluster_names[cluster] = "Champions"
        elif recency < rfm['Recency'].quantile(0.5) and frequency > rfm['Frequency'].quantile(0.5):
            cluster_names[cluster] = "Loyal Customers"
        elif recency < rfm['Recency'].quantile(0.75) and frequency > rfm['Frequency'].quantile(0.25):
            cluster_names[cluster] = "Potential Loyalists"
        else:
            cluster_names[cluster] = "At Risk"
    rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_names)
    return rfm

# Download CSV
def download_csv(df, filename="rfm_clusters.csv"):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return st.download_button(
        label="📥 Download as CSV",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=str(uuid.uuid4())
    )

# Download Excel
def download_excel(df, filename="rfm_clusters.xlsx"):
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    return st.download_button(
        label="📥 Download as Excel",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=str(uuid.uuid4())
    )

# Generate PDF Report
def generate_pdf_report(rfm, cluster_summary):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Customer Segmentation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, "Generated on: April 24, 2025")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 680, "Key Insights")
    c.setFont("Helvetica", 12)
    y = 660
    insights = [
        f"Total Customers: {len(rfm)}",
        f"Number of Clusters: {len(cluster_summary)}",
        f"Average Recency: {rfm['Recency'].mean():.1f} days",
        f"Average Frequency: {rfm['Frequency'].mean():.1f} orders",
        f"Average Monetary: £{rfm['Monetary'].mean():.2f}"
    ]
    for insight in insights:
        c.drawString(60, y, f"• {insight}")
        y -= 20
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y-20, "Cluster Summary")
    c.setFont("Helvetica", 12)
    y -= 40
    for _, row in cluster_summary.iterrows():
        c.drawString(60, y, f"Cluster {row['Cluster_Name']}: {row['Customer Count']} customers, "
                           f"Recency: {row['Recency']:.1f}, Frequency: {row['Frequency']:.1f}, Monetary: £{row['Monetary']:.2f}")
        y -= 20
    
    c.showPage()
    c.save()
    return buffer.getvalue()

# Sidebar
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    uploaded = st.file_uploader("📂 Upload OnlineRetail.xlsx", type=["xlsx"])
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4, key="num_clusters")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This dashboard uses RFM analysis and KMeans clustering to segment customers, providing actionable marketing insights.")

# Main App
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("**Unlock customer insights with RFM-based clustering for targeted marketing strategies.**")

if uploaded:
    with st.container():
        st.subheader("📊 Key Insights")
        df = load_data(uploaded)
        df = preprocess_data(df)
        rfm = create_rfm(df)
        rfm = perform_kmeans(rfm, k=num_clusters)
        rfm = name_clusters(rfm)

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(rfm))
        col2.metric("Avg Recency", f"{rfm['Recency'].mean():.1f} days")
        col3.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
        col4.metric("Avg Monetary", f"£{rfm['Monetary'].mean():.2f}")

        # Cluster Summary
        cluster_summary = rfm.groupby(['Cluster', 'Cluster_Name']).agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "CustomerID": "count"
        }).round(1).reset_index()
        cluster_summary.rename(columns={"CustomerID": "Customer Count"}, inplace=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Customer Data", "📊 Cluster Analysis", "📈 Visualizations"])

        with tab1:
            st.subheader("📋 Customer RFM Data")
            cluster_filter = st.multiselect("Filter by Cluster Name", options=rfm['Cluster_Name'].unique(), default=rfm['Cluster_Name'].unique())
            filtered_rfm = rfm[rfm['Cluster_Name'].isin(cluster_filter)]
            st.dataframe(filtered_rfm.sort_values(by="Cluster"), height=300, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                download_csv(filtered_rfm)
            with col2:
                download_excel(filtered_rfm)

        with tab2:
            st.subheader("📊 Cluster Profiles")
            # Bar Plot
            melted_summary = cluster_summary.melt(
                id_vars=["Cluster", "Cluster_Name"],
                value_vars=["Recency", "Frequency", "Monetary"],
                var_name="Metric",
                value_name="Mean Value"
            )
            bar_fig = px.bar(
                melted_summary,
                x="Metric",
                y="Mean Value",
                color="Cluster_Name",
                barmode="group",
                text="Mean Value",
                title="📊 Average RFM Scores by Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            bar_fig.update_traces(
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f} (%{customdata})",
                customdata=melted_summary["Cluster_Name"]
            )
            bar_fig.update_layout(
                yaxis_title="Average Value",
                xaxis_title="RFM Metric",
                hovermode="closest",
                legend_title="Cluster"
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
                color="Cluster_Name",
                size="Frequency",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary", "Cluster_Name"],
                title="🧭 RFM Cluster Distribution (Recency vs Monetary)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            scatter_fig.update_traces(
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="CustomerID: %{customdata[0]}<br>Recency: %{x}<br>Monetary: %{y}<br>Frequency: %{customdata[2]}<br>Cluster: %{customdata[4]}"
            )
            scatter_fig.update_layout(
                xaxis_title="Recency (days)",
                yaxis_title="Monetary Value (£)",
                hovermode="closest",
                legend_title="Cluster"
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Business Insights
        with st.expander("💡 Marketing Strategies by Cluster", expanded=True):
            st.markdown("""
            **Tailored Strategies**:
            - **Champions**: Exclusive rewards, VIP programs, early product access.
            - **Loyal Customers**: Referral incentives, loyalty discounts.
            - **Potential Loyalists**: Personalized follow-ups, cross-sell opportunities.
            - **At Risk**: Win-back campaigns, surveys to understand churn.
            """)

        # Download PDF Report
        st.subheader("📑 Export Report")
        pdf_data = generate_pdf_report(rfm, cluster_summary)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name="customer_segmentation_report.pdf",
            mime="application/pdf",
            key=str(uuid.uuid4())
        )

else:
    st.warning("📁 Please upload the `OnlineRetail.xlsx` dataset to begin.")
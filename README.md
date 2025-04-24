# 🛍️ Customer Segmentation Using Clustering Techniques

Identify distinct customer groups based on purchasing behavior to help businesses personalize marketing strategies.

## 🔍 Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Description**: Transaction data from a UK-based online retailer (2010–2011).

## 🚀 Features
- Upload and process `.xlsx` E-commerce datasets
- Compute RFM metrics (Recency, Frequency, Monetary)
- Run KMeans clustering
- Visualize clusters using PCA and t-SNE
- Download clustered customer segments

## 📊 Tech Stack
- Python 🐍 (Pandas, Scikit-learn, Seaborn, Matplotlib)
- Streamlit 🖥️ for web app
- Excel support via OpenPyXL

## 🧠 Key Steps

### 1. Data Preprocessing
- Handle missing values
- Remove canceled orders
- Create TotalPrice and RFM features

### 2. Clustering Techniques
- KMeans with 4 clusters
- t-SNE & PCA for visualization

### 3. Cluster Interpretation
- Champions: Exclusive offers
- Loyal: Encourage referrals
- Potential Loyalists: Personalized upsell
- At Risk: Win-back campaigns

## 📦 Setup

```bash
pip install -r requirements.txt
streamlit run app.py

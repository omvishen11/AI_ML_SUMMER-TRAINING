import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Data Loading
# ==========================================
@st.cache_data
def load_data(filepath="income.csv"):
    """Safely loads and caches the CSV file."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset '{filepath}' not found. Please place it in the same directory.")
        return None

df = load_data()

# Stop execution if data isn't found
if df is None:
    st.stop()

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ==========================================
# 3. Sidebar UI (Reactive)
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Feature Selection
    default_features = [col for col in ["Age", "Income($)"] if col in numerical_cols]
    if len(default_features) < 2 and len(numerical_cols) >= 2:
        default_features = numerical_cols[:2]
        
    selected_features = st.multiselect(
        "Select features for clustering:",
        options=numerical_cols,
        default=default_features
    )
    
    # Slider for K
    k_value = st.slider("Number of Clusters (K):", min_value=2, max_value=10, value=3, step=1)
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 numerical features to run clustering.")
        st.stop()

# ==========================================
# 4. Clustering Logic
# ==========================================
# Scale features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[selected_features] = scaler.fit_transform(df[selected_features])

# Run K-Means
km = KMeans(n_clusters=k_value, random_state=42)
df['Cluster'] = km.fit_predict(df_scaled[selected_features])
df_scaled['Cluster'] = df['Cluster']
centroids = km.cluster_centers_

# ==========================================
# 5. Main Dashboard Layout
# ==========================================
st.title("📊 K-Means Clustering App")
st.write("Analyze, segment, and visualize customer demographic data interactively.")
st.divider()

# Create Tabs
tab_preview, tab_analysis, tab_elbow = st.tabs([
    "📂 Dataset Insights", 
    "🎯 Clustering Analysis", 
    "📈 Elbow Method"
])

# ------------------- TAB 1: DATASET PREVIEW -------------------
with tab_preview:
    st.subheader("Summary Statistics")
    
    # Native Streamlit Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.markdown("**Raw Dataset Preview**")
        st.dataframe(df.drop(columns=['Cluster']), use_container_width=True)
        
    with col_right:
        st.markdown("**Descriptive Stats**")
        st.dataframe(df.describe().T[["mean", "std", "min", "max"]], use_container_width=True)

# ------------------- TAB 2: CLUSTERING ANALYSIS -------------------
with tab_analysis:
    st.subheader(f"Clustering Results (K = {k_value})")
    
    col_plot, col_table = st.columns([3, 2], gap="large")
    
    with col_plot:
        st.markdown(f"**Scatter Plot: {selected_features[0]} vs {selected_features[1]}**")
        st.caption("Coordinates are normalized [0, 1] for K-Means accuracy.")
        
        # Matplotlib for scatter + centroids
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('transparent') # Looks great in dark/light mode
        
        cmap = plt.get_cmap('tab10')
        for i in range(k_value):
            cluster_data = df_scaled[df_scaled['Cluster'] == i]
            ax.scatter(cluster_data[selected_features[0]], cluster_data[selected_features[1]], 
                       color=cmap(i), label=f'Cluster {i}', s=50, alpha=0.7)
            
        # Plot Centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], color='purple', marker='*', s=250, label='Centroids', edgecolor='black')
        
        ax.set_xlabel(f'Scaled {selected_features[0]}')
        ax.set_ylabel(f'Scaled {selected_features[1]}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
    with col_table:
        st.markdown("**Clustered Output Data**")
        st.dataframe(df, use_container_width=True)
        
        st.download_button(
            label="⬇️ Download Clustered Dataset",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="clustered_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("**Scaled Centroids**")
        st.dataframe(pd.DataFrame(centroids, columns=[f"Scaled {f}" for f in selected_features]), use_container_width=True)

# ------------------- TAB 3: ELBOW METHOD -------------------
with tab_elbow:
    st.subheader("Elbow Curve Optimization")
    st.write("The Elbow Method computes the sum of squared errors (SSE) for different values of K.")
    
    # Calculate SSE for k=1 to 9
    sse = []
    k_range = range(1, 10)
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42)
        km_temp.fit(df_scaled[selected_features])
        sse.append(km_temp.inertia_)
        
    col_elbow_plot, col_elbow_desc = st.columns([3, 2], gap="large")
    
    with col_elbow_plot:
        # Replaced matplotlib with Streamlit's native interactive line chart
        chart_data = pd.DataFrame({"K (Number of Clusters)": k_range, "SSE": sse}).set_index("K (Number of Clusters)")
        st.line_chart(chart_data)
        
    with col_elbow_desc:
        st.info("""
        **How to interpret this curve:**
        * **SSE** decreases as the number of clusters (K) increases.
        * Look for the **'elbow'** point where the line stops dropping sharply and levels out.
        * In most datasets, the elbow point indicates the optimal number of clusters to use.
        """)
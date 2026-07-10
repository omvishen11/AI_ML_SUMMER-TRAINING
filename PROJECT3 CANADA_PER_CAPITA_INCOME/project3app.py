

import streamlit as st
import pandas as pd

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Outlier Detection Using Percentile",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------
# Title
# ---------------------------------------
st.title("📊 Outlier Detection Using Percentile Method")
st.write(
    """
    This application detects and removes outliers from the
    **Per Capita Income** column using the Percentile Method.
    """
)

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("canada_per_capita_income.csv")   # Change filename if required

# ---------------------------------------
# Display Dataset
# ---------------------------------------
st.header("📋 Original Dataset")

st.dataframe(df)

st.write("Dataset Shape:", df.shape)

# ---------------------------------------
# Calculate Percentiles
# ---------------------------------------
st.header("📊 Percentile Calculation")

min_threshold, max_threshold = df["per capita income (US$)"].quantile([0.01, 0.999])

col1, col2 = st.columns(2)

with col1:
    st.metric("Minimum Threshold", round(min_threshold,2))

with col2:
    st.metric("Maximum Threshold", round(max_threshold,2))

# ---------------------------------------
# Show Lower Outliers
# ---------------------------------------
st.header("📉 Lower Outliers")

lower_outliers = df[df["per capita income (US$)"] < min_threshold]

st.write("Number of Lower Outliers:", lower_outliers.shape[0])

st.dataframe(lower_outliers)

# ---------------------------------------
# Remove Outliers
# ---------------------------------------
df2 = df[
    (df["per capita income (US$)"] > min_threshold) &
    (df["per capita income (US$)"] < max_threshold)
]

# ---------------------------------------
# Cleaned Dataset
# ---------------------------------------
st.header("✅ Dataset After Removing Outliers")

st.write("Shape:", df2.shape)

st.dataframe(df2.sample(5))

# ---------------------------------------
# Summary Statistics
# ---------------------------------------
st.header("📈 Summary Statistics")

st.dataframe(df2["per capita income (US$)"].describe())

# ---------------------------------------
# Comparison
# ---------------------------------------
st.header("📊 Dataset Comparison")

comparison = pd.DataFrame({
    "Dataset": ["Original", "After Removing Outliers"],
    "Rows": [df.shape[0], df2.shape[0]],
    "Columns": [df.shape[1], df2.shape[1]]
})

st.dataframe(comparison)

st.success("✅ Outliers removed successfully using the Percentile Method.")

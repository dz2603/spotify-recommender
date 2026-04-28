import numpy as np
import pandas as pd
import streamlit as st

from src.ui.common import load_data


def page_dataset_info():
    st.header("🗂️ Dataset Information")
    st.markdown("Quick overview of the dataset used across all app pages.")

    data, X_num, feat_cols, _ = load_data(sample_n=None)

    n_rows, n_cols = data.shape
    n_numeric = len(data.select_dtypes(include=[np.number]).columns)
    n_non_numeric = n_cols - n_numeric
    n_missing = int(data.isna().sum().sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{n_rows:,}")
    m2.metric("Columns", f"{n_cols:,}")
    m3.metric("Missing Values", f"{n_missing:,}")
    m4.metric("Numeric Features", f"{n_numeric:,}")

    st.markdown("---")
    st.subheader("Column Summary")
    summary_df = pd.DataFrame(
        {
            "Column": data.columns,
            "Dtype": [str(dtype) for dtype in data.dtypes],
            "Non-Null Count": [int(data[col].notna().sum()) for col in data.columns],
            "Missing Count": [int(data[col].isna().sum()) for col in data.columns],
        }
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Rows")
    st.dataframe(data.head(20), hide_index=True, use_container_width=True)

    with st.expander("Feature Matrix Details"):
        st.markdown(
            f"- Standardized numeric matrix shape: **{X_num.shape[0]:,} x {X_num.shape[1]:,}**\n"
            f"- Numeric feature count used in modeling: **{len(feat_cols):,}**\n"
            f"- Non-numeric columns in raw data: **{n_non_numeric:,}**"
        )

    st.success(
        "✅ **Feature Engineering Status**: All numeric data has been successfully standardized (Mean=0, Std=1). "
        "This is a critical step for distance-based algorithms like KNN and K-Means to ensure mathematically sound results."
    )

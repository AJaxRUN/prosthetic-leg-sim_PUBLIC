import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV from the specified folder
FILE_PATH = "data/plot_data.csv"

try:
    df = pd.read_csv(FILE_PATH)
    st.title("Dynamic Multi-Y Axis Plot from CSV")

    # Select columns for X and multiple Y values
    x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
    y_cols = st.sidebar.multiselect("Select Y-axis column(s)", df.columns)

    # Ensure user selects at least one Y column
    if y_cols:
        # Plot the data
        fig, ax = plt.subplots()
        for y_col in y_cols:
            ax.plot(df[x_col], df[y_col], label=y_col)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel("Y Values")
        ax.legend(loc="upper left")
        st.pyplot(fig)
    else:
        st.warning("Please select at least one Y-axis column.")
except FileNotFoundError:
    st.error(f"File not found: {FILE_PATH}. Please ensure the file exists.")
except Exception as e:
    st.error(f"An error occurred: {e}")

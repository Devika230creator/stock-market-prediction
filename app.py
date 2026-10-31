
import streamlit as st
import pandas as pd
import joblib
import os

# PAGE SETTINGS
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# TITLE
st.title("📈 Stock Market Prediction Using Machine Learning")
st.write(
    "A machine learning system to analyze historical "
    "stock data and predict possible market movements."
)

st.info(
    "Educational project only. Predictions are not "
    "financial or investment advice."
)

# LOAD DATA
if not os.path.exists("prepared_stock_data.csv"):
    st.error("Stock dataset is not available.")
    st.stop()

df = pd.read_csv("prepared_stock_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# SIDEBAR
st.sidebar.title("Project Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Dashboard", "Stock Data", "Model Comparison", "Prediction"]
)

# DASHBOARD
if page == "Dashboard":

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Stock Company", "Reliance Industries")
    col2.metric("Data Records", len(df))
    col3.metric("ML Algorithms", "2")

    st.subheader("Historical Stock Price")

    chart_data = df.set_index("Date")[["Close"]]
    st.line_chart(chart_data)

    st.subheader("Recent Stock Data")
    st.dataframe(df.tail(10), use_container_width=True)

# STOCK DATA
elif page == "Stock Data":

    st.subheader("Historical Stock Market Data")

    st.write("Open, High, Low, Close and Trading Volume")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Stock Dataset",
        df.to_csv(index=False),
        file_name="reliance_stock_data.csv",
        mime="text/csv"
    )

# MODEL COMPARISON
elif page == "Model Comparison":

    st.subheader("Machine Learning Model Comparison")

    st.write("Initial experimental results")

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy (%)": [53.85, 51.42],
        "Precision (%)": [50.34, 48.12],
        "Recall (%)": [65.22, 55.65],
        "F1 Score (%)": [56.82, 51.61]
    })

    st.dataframe(comparison, use_container_width=True)

    st.bar_chart(
        comparison.set_index("Model")[["Accuracy (%)"]]
    )

    st.success(
        "Logistic Regression performed better "
        "in our initial experiment."
    )

# PREDICTION
elif page == "Prediction":

    st.subheader("Next-Day Stock Movement Prediction")

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        ["Logistic Regression", "Random Forest"]
    )

    model_files = {
        "Logistic Regression": "logistic_model.pkl",
        "Random Forest": "random_forest_model.pkl"
    }

    st.write(
        "Select the model and predict the next trading "
        "day's possible stock movement."
    )

    if st.button("Predict Stock Movement"):

        model_file = model_files[model_choice]

        if not os.path.exists(model_file):
            st.warning(
                "The trained model file has not been uploaded yet."
            )
        else:
            model = joblib.load(model_file)

            features = [
                ["Open", "High", "Low", "Close", "Volume"]
            ]

            latest = df[features[0]].iloc[[-1]]

            prediction = model.predict(latest)[0]

            if prediction == 1:
                st.success("Predicted Movement: UP 📈")
            else:
                st.error("Predicted Movement: DOWN 📉")

            st.caption(
                "Prediction is based on the latest available "
                "completed trading day's data."
            )

# FOOTER
st.divider()

st.caption(
    "Developed as an academic project: "
    "Stock Market Prediction Using Machine Learning."
)

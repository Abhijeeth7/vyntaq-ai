import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from dotenv import load_dotenv

# ---------------- LOAD ENV ---------------- #
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Vyntaq")
st.title("Vyntaq AI")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# ---------------- MAIN LOGIC ---------------- #
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # 🔥 CLEAN DATA
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors='coerce')

    # ---------------- PREVIEW ---------------- #
    st.subheader("Data Preview")
    st.dataframe(df)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

    # ---------------- VISUALIZATION ---------------- #
    st.subheader("📊 Interactive Visualization")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_columns) == 0:
        st.warning("No numeric columns available")

    else:
        selected_column = st.selectbox(
            "Select metric to analyze",
            numeric_columns
        )

        if "Rank" in df.columns:
            x_axis = "Rank"
        else:
            df = df.reset_index()
            x_axis = "index"

        plot_df = df[[x_axis, selected_column]].dropna()
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
        plot_df = plot_df.sort_values(by=x_axis)

        # ---------------- PREVIEW ---------------- #
        st.write("🔍 Data Preview:")
        st.dataframe(plot_df.head(10))

        # ---------------- CHART ---------------- #
        fig = px.line(
            plot_df,
            x=x_axis,
            y=selected_column,
            title=f"{selected_column} vs {x_axis}",
            markers=True,
            log_y=True if selected_column == "Reviews" else False
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- AI INSIGHTS ---------------- #
        st.subheader("🧠 AI Insights")

        if st.button("Generate AI Insights"):

            summary = plot_df.describe().to_string()

            prompt = f"""
            You are a business data analyst.

            Dataset summary:
            {summary}

            Give meaningful business insights (trends, anomalies, recommendations).
            Avoid generic statements.
            """

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}"
                    },
                    json={
                        "model": "openai/gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                result = response.json()

                if "choices" in result:
                    st.success("AI Insights:")
                    st.write(result['choices'][0]['message']['content'])
                else:
                    st.error("API Error")
                    st.write(result)

            except Exception as e:
                st.error("Something went wrong")
                st.write(str(e))
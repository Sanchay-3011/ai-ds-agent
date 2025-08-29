import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from chatbot import chatbot_sidebar

st.session_state["page_name"] = "Report"

st.title("📑 Generate Report")

# -------------------------
# Load dataset from session_state
# -------------------------
if "dataset" in st.session_state and "uploaded_filename" in st.session_state:
    df = st.session_state["dataset"]
    dataset_name = st.session_state["uploaded_filename"].split(".")[0]

    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # -------------------------
        # Title
        # -------------------------
        elements.append(Paragraph(f"{dataset_name} Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # -------------------------
        # Dataset Overview
        # -------------------------
        elements.append(Paragraph("📊 Dataset Overview", styles['Heading2']))
        elements.append(Paragraph(f"Rows: {df.shape[0]}", styles['Normal']))
        elements.append(Paragraph(f"Columns: {df.shape[1]}", styles['Normal']))
        elements.append(Paragraph(f"Missing Values: {df.isnull().sum().sum()}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # -------------------------
        # Descriptive Statistics (Split into chunks so it fits PDF)
        # -------------------------
        elements.append(Paragraph("📈 Descriptive Statistics", styles['Heading2']))

        stats_df = df.describe().round(2).reset_index()

        # Format numbers with commas + 2 decimals
        def format_value(x):
            if isinstance(x, (int, float)):
                return f"{x:,.2f}"
            return str(x)

        stats_df = stats_df.applymap(format_value)

        chunk_size = 6  # number of columns per table
        for start in range(0, stats_df.shape[1], chunk_size):
            subset = stats_df.iloc[:, start:start + chunk_size]
            table_data = [subset.columns.tolist()] + subset.values.tolist()

            stats_table = Table(table_data, repeatRows=1)
            style_commands = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 7),  # smaller font
            ]
            # Alternating row background
            for i in range(1, len(table_data)):
                bg_color = colors.whitesmoke if i % 2 == 0 else colors.lightgrey
                style_commands.append(('BACKGROUND', (0, i), (-1, i), bg_color))

            stats_table.setStyle(TableStyle(style_commands))
            elements.append(stats_table)
            elements.append(Spacer(1, 12))

        # -------------------------
        # Best Model Summary
        # -------------------------
        if "best_model" in st.session_state:
            elements.append(Paragraph("🤖 Best Model Summary", styles['Heading2']))
            model_table_data = [
                ["Model", "Score", "Type"],
                [
                    st.session_state["best_model_name"],
                    f"{st.session_state['best_score']:.4f}",
                    st.session_state["problem_type"]
                ]
            ]
            model_table = Table(model_table_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2196F3")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(model_table)
            elements.append(Spacer(1, 12))

        # -------------------------
        # Correlation Heatmap (if numeric)
        # -------------------------
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close(fig)
            img_buffer.seek(0)
            elements.append(Paragraph("📌 Correlation Heatmap", styles['Heading2']))
            elements.append(Image(img_buffer, width=400, height=300))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()

    st.download_button(
        label="📥 Download Detailed Report (PDF)",
        data=pdf_buffer,
        file_name=f"{dataset_name}_report.pdf",
        mime="application/pdf"
    )

else:
    st.warning("⚠️ Please upload and process a dataset first.")

# Chatbot
chatbot_sidebar()

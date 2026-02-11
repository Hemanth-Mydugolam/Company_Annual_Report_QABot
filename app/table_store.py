import streamlit as st
from rag import answer_question
import json

st.set_page_config(page_title="Company Intelligence AI", layout="wide")

st.title("📊 Company Document Intelligence")

with open("parsed_data.json") as f:
    companies = list(json.load(f).keys())

company = st.selectbox("Select Company", companies)
question = st.text_input("Ask a question about the company")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        answer = answer_question(question, company)
        st.markdown("### Answer")
        st.write(answer)

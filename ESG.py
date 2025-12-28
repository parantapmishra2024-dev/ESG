import streamlit as st
import pandas as pd
import PyPDF2
from google import genai
from google.genai import types

# Page Configuration
st.set_page_config(page_title="Berlin Packaging ESG Researcher", layout="wide")

st.title("Berlin Packaging ESG Research Tool")
st.markdown("Upload your questions and report to get AI-powered ESG analysis with live web grounding.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    limit = st.number_input("Question Limit (0 for all)", min_value=0, value=5)
    st.info("The tool prioritizes the uploaded PDF, then falls back to Google Search for the most current data.")

# --- File Uploads ---
col1, col2 = st.columns(2)
with col1:
    questions_file = st.file_uploader("Upload Questions (Excel)", type=["xlsx"])
with col2:
    report_file = st.file_uploader("Upload ESG Report (PDF) - Optional", type=["pdf"])

# --- Helper Functions ---
def extract_text_from_pdf(file):
    if not file:
        return None
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except:
        return None

def get_answer_with_sources(client, question, report_text=None):
    prompt = f"""
You are an ESG and sustainability analyst preparing a factual, auditable response.

Instructions:
1. Use the provided company report context FIRST.
2. Only if the answer is not found in the report, use Google Search to find current, credible public sources.
3. Do NOT make assumptions or speculate.
4. If the information is not available in either source, explicitly say: "Information not publicly disclosed."
5. Keep the answer concise, factual, and formal.
6. Prefer numeric values, dates, and policy names where applicable.
Avoid marketing or vague language.

Company: Berlin Packaging

Company Report Context:
{report_text[:25000] if report_text else "No internal report provided."}

Question:
{question}

Answer:
"""

    search_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(
        tools=[search_tool],
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=512,
        response_mime_type="text/plain"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )

        answer_text = response.text.strip()
        sources = []

        if hasattr(response, 'candidates') and response.candidates[0].grounding_metadata:
            chunks = response.candidates[0].grounding_metadata.grounding_chunks
            for chunk in chunks:
                if chunk.web and chunk.web.uri:
                    sources.append(chunk.web.uri)

        source_links = "\n".join(list(set(sources))) if sources else "Local Report / General Knowledge"
        return answer_text, source_links

    except Exception as e:
        return f"Error: {str(e)}", "N/A"

# --- Main Processing ---
if st.button("Start Research") and questions_file and api_key:
    client = genai.Client(api_key=api_key)

    df_questions = pd.read_excel(questions_file)
    q_col = 'Question' if 'Question' in df_questions.columns else df_questions.columns[0]
    questions_list = df_questions[q_col].dropna().tolist()

    if limit > 0:
        questions_list = questions_list[:limit]

    report_text = extract_text_from_pdf(report_file)
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, q in enumerate(questions_list, 1):
        status_text.text(f"Processing question {i} of {len(questions_list)}...")
        ans, src = get_answer_with_sources(client, q, report_text)
        results.append({"Question": q, "Answer": ans, "Sources": src})
        progress_bar.progress(i / len(questions_list))

    # --- Display Results ---
    st.success("Analysis Complete!")
    df_results = pd.DataFrame(results)

    def format_sources(src):
        if src == "Local Report / General Knowledge":
            return src
        return "  \n".join([f"[{url.split('//')[-1].split('/')[0]}]({url})" for url in src.split("\n")])

    df_display = df_results.copy()
    df_display['Sources'] = df_display['Sources'].apply(format_sources)

    st.dataframe(df_display, use_container_width=True)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Results (CSV)", csv, "Berlin_Packaging_ESG_Results.csv", "text/csv")

elif not api_key and questions_file:
    st.warning("Please enter your API Key in the sidebar.")

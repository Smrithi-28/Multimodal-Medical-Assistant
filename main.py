from pathlib import Path
from functools import lru_cache
import os
import html

import gradio as gr
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from brain_of_the_doctor_groq import brain_of_the_doctor
from voice_of_the_doctor import convert_text_to_doctor_audio
from voice_of_the_patient import transcribe_patient_voice


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

APP_TITLE = "Multimodal Medical Assistant"

VECTORSTORE_PATH = BASE_DIR / "vectorstore" / "db_faiss"


# ============================================================
# GLOBAL CSS
# ============================================================

CSS = """
/* ============================================================
   MULTIMODAL MEDICAL ASSISTANT — CLEAN LIGHT UI
   ============================================================ */

:root {
    --mma-bg: #f4f7fb;
    --mma-surface: #ffffff;
    --mma-surface-soft: #f8fafc;
    --mma-primary: #2563eb;
    --mma-primary-dark: #1d4ed8;
    --mma-primary-soft: #eff6ff;
    --mma-border: #dbe3ef;
    --mma-border-soft: #e8edf4;
    --mma-text: #172033;
    --mma-text-soft: #344054;
    --mma-muted: #667085;
    --mma-radius: 18px;

    --body-background-fill: #f4f7fb;
    --body-text-color: #172033;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f8fafc;
    --block-background-fill: #ffffff;
    --block-border-color: #dbe3ef;
    --input-background-fill: #f8fafc;
    --input-background-fill-focus: #ffffff;
    --input-border-color: #dbe3ef;
    --input-border-color-focus: #2563eb;
    --button-primary-background-fill: #2563eb;
    --button-primary-background-fill-hover: #1d4ed8;
    --button-primary-text-color: #ffffff;

    color-scheme: light;
}


/* -------------------- PAGE -------------------- */

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--mma-bg) !important;
    color: var(--mma-text) !important;
}

body,
.gradio-container,
.gradio-container * {
    box-sizing: border-box !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100vh !important;
    background: var(--mma-bg) !important;
    color: var(--mma-text) !important;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif !important;

    color-scheme: light !important;
}


/* -------------------- MAIN SHELL -------------------- */

.mma-shell {
    width: min(1180px, calc(100% - 32px)) !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 0 40px !important;
    overflow: visible !important;
}

.mma-shell .gr-row,
.mma-shell .gr-column,
.mma-shell .block,
.mma-shell .form,
.mma-shell .wrap {
    min-width: 0 !important;
}

.mma-shell .gr-row {
    width: 100% !important;
}


/* -------------------- HEADER -------------------- */

.mma-header {
    width: 100% !important;
    min-height: 118px;
    background: #ffffff !important;
    border: 1px solid var(--mma-border) !important;
    border-radius: 22px !important;
    padding: 22px 28px !important;
    margin: 0 0 26px !important;
    box-shadow: 0 10px 30px rgba(16, 24, 40, .06) !important;
}

.mma-header-content {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 24px !important;
    width: 100% !important;
}

.mma-brand {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    min-width: 0 !important;
}

.mma-brand-icon {
    width: 58px !important;
    height: 58px !important;
    flex: 0 0 58px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 16px !important;
    background: var(--mma-primary-soft) !important;
    color: var(--mma-primary) !important;
    font-size: 30px !important;
    line-height: 1 !important;
}

.mma-brand h1 {
    margin: 0 !important;
    color: var(--mma-text) !important;
    font-size: 29px !important;
    line-height: 36px !important;
    font-weight: 750 !important;
    letter-spacing: -.025em !important;
}

.mma-brand p {
    margin: 4px 0 0 !important;
    color: var(--mma-muted) !important;
    font-size: 11px !important;
    line-height: 17px !important;
    font-weight: 700 !important;
    letter-spacing: .06em !important;
}

.mma-security {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex: 0 0 auto !important;
    color: #000000 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

.mma-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: var(--mma-primary) !important;
    font-family: Arial, sans-serif !important;
    font-size: 25px !important;
    line-height: 1 !important;
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}

.mma-brand-icon {
    font-size: 0 !important;
}

.mma-brand-icon::before {
    content: "⚕";
    font-size: 32px;
}

.mma-security .mma-icon {
    font-size: 0 !important;
}

.mma-security .mma-icon::before {
    content: "✓";
    font-size: 18px;
    font-weight: 800;
}


/* -------------------- INTRO -------------------- */

.mma-intro {
    text-align: center !important;
    padding: 4px 16px 26px !important;
}

.mma-intro h2 {
    margin: 0 0 8px !important;
    color: var(--mma-text) !important;
    font-size: 27px !important;
    line-height: 34px !important;
    font-weight: 750 !important;
}

.mma-intro p {
    max-width: 850px !important;
    margin: 0 auto !important;
    color: var(--mma-muted) !important;
    font-size: 14px !important;
    line-height: 21px !important;
}


/* -------------------- TABS -------------------- */

.mma-tabs,
.mma-tabs > div {
    width: 100% !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.mma-tabs .tab-nav {
    display: flex !important;
    align-items: stretch !important;
    gap: 5px !important;
    width: 100% !important;
    min-height: 54px !important;
    padding: 5px !important;
    margin: 0 0 22px !important;
    background: #ffffff !important;
    border: 1px solid var(--mma-border) !important;
    border-radius: 15px !important;
    box-shadow: 0 5px 18px rgba(16, 24, 40, .045) !important;
}

.mma-tabs .tab-nav button,
.mma-tabs .tab-nav button[role="tab"] {
    flex: 0 1 auto !important;
    min-height: 42px !important;
    margin: 0 !important;
    padding: 10px 20px !important;
    background: #f3f5f8 !important;
    border: 1px solid #d0d5dd !important;
    border-radius: 10px !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    line-height: 20px !important;
    white-space: nowrap !important;
}


/* Tab text */

.mma-tabs .tab-nav button span,
.mma-tabs .tab-nav button div,
.mma-tabs .tab-nav button p {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}


/* Hover */

.mma-tabs .tab-nav button:hover,
.mma-tabs .tab-nav button:focus-visible {
    background: #eaf1ff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border-color: #cbdcff !important;
    opacity: 1 !important;
}


/* Selected */

.mma-tabs .tab-nav button.selected,
.mma-tabs .tab-nav button[aria-selected="true"] {
    background: #2563eb !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border-color: #2563eb !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, .20) !important;
}


/* -------------------- MODE INFO -------------------- */

.mma-mode-info {
    display: flex !important;
    align-items: flex-start !important;
    gap: 13px !important;
    width: 100% !important;
    padding: 15px 17px !important;
    margin: 0 0 20px !important;
    background: var(--mma-primary-soft) !important;
    border: 1px solid #cfe0ff !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

.mma-mode-info .mma-icon {
    width: 28px !important;
    font-size: 23px !important;
}

.mma-mode-info strong {
    display: block !important;
    margin: 0 0 3px !important;
    color: var(--mma-text) !important;
    font-size: 14px !important;
    line-height: 20px !important;
    font-weight: 750 !important;
}

.mma-mode-info span:not(.mma-icon) {
    display: block !important;
    color: var(--mma-text-soft) !important;
    font-size: 12px !important;
    line-height: 18px !important;
}


/* -------------------- SECTION HEADINGS -------------------- */

.mma-section-title {
    display: flex !important;
    align-items: center !important;
    gap: 9px !important;
    min-width: 0 !important;
    margin: 0 0 13px !important;
}

.mma-section-title h2 {
    margin: 0 !important;
    color: var(--mma-text) !important;
    font-size: 21px !important;
    line-height: 28px !important;
    font-weight: 750 !important;
}

.mma-section-title .mma-icon {
    width: 27px !important;
    font-size: 22px !important;
    line-height: 1 !important;
}


/* -------------------- CARDS -------------------- */

.mma-card {
    width: 100% !important;
    min-width: 0 !important;
    background: #ffffff !important;
    border: 1px solid var(--mma-border) !important;
    border-radius: var(--mma-radius) !important;
    box-shadow: 0 8px 26px rgba(16, 24, 40, .055) !important;
    padding: 20px !important;
    overflow: visible !important;
}

.mma-card .wrap,
.mma-card .block,
.mma-card .form {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}


/* -------------------- GENERAL TEXT -------------------- */

.mma-card label,
.mma-card .label-wrap,
.mma-card label span {
    color: var(--mma-text-soft) !important;
    -webkit-text-fill-color: var(--mma-text-soft) !important;
    background: transparent !important;
    opacity: 1 !important;
}

.mma-card label span {
    font-size: 11px !important;
    line-height: 17px !important;
    font-weight: 750 !important;
    letter-spacing: .055em !important;
    text-transform: uppercase !important;
}


/* -------------------- INPUTS -------------------- */

.mma-card input,
.mma-card textarea,
.mma-card select,
.mma-output textarea,
.mma-chat-input textarea {
    background: #f8fafc !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

.mma-card input:focus,
.mma-card textarea:focus,
.mma-card select:focus,
.mma-output textarea:focus,
.mma-chat-input textarea:focus {
    background: #ffffff !important;
    border-color: #2563eb !important;
    box-shadow:
        0 0 0 3px rgba(37, 99, 235, .10) !important;
}

.mma-card textarea,
.mma-output textarea {
    line-height: 22px !important;
}

.mma-card ::placeholder {
    color: #98a2b3 !important;
    -webkit-text-fill-color: #98a2b3 !important;
    opacity: 1 !important;
}


/* -------------------- CHATBOT -------------------- */

.mma-chatbot {
    width: 100% !important;
    min-width: 0 !important;
    height: 450px !important;
    background: #f8fafc !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    color: #172033 !important;
    color-scheme: light !important;
}

.mma-chatbot > div,
.mma-chatbot [class*="message"],
.mma-chatbot [class*="chatbot"],
.mma-chatbot .wrap,
.mma-chatbot .container,
.mma-chatbot .prose,
.mma-chatbot .bubble-wrap {
    background: #f8fafc !important;
    color: #172033 !important;
}

.mma-chatbot .message,
.mma-chatbot p,
.mma-chatbot span,
.mma-chatbot div {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    overflow-wrap: anywhere !important;
}

.mma-chatbot [data-testid="bot"],
.mma-chatbot [data-testid="bot"] * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

.mma-chatbot pre,
.mma-chatbot code {
    background: #eef2f7 !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}


/* -------------------- BUTTONS -------------------- */

.mma-card button,
.mma-card .gr-button {
    opacity: 1 !important;
    text-shadow: none !important;
}


/* PRIMARY BUTTON */

.mma-primary-button .gr-button-primary,
.mma-card .gr-button-primary {
    min-height: 50px !important;
    padding: 11px 18px !important;
    background: #2563eb !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 15px rgba(37, 99, 235, .18) !important;
}

.mma-primary-button .gr-button-primary:hover,
.mma-card .gr-button-primary:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}


/* SECONDARY BUTTON */

.mma-card .gr-button-secondary {
    min-height: 42px !important;
    background: #f3f5f8 !important;
    color: #344054 !important;
    -webkit-text-fill-color: #344054 !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
}

.mma-card .gr-button-secondary:hover {
    background: #eaf1ff !important;
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
}


/* -------------------- RECORD BUTTON -------------------- */

.mma-card .gradio-audio button[aria-label="Record"],
.mma-card .gradio-audio button[aria-label*="Record"],
.mma-card .gradio-audio button[title*="Record"],
.mma-card .gradio-audio [data-testid="record"] {
    background: #1f2937 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #1f2937 !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

.mma-card .gradio-audio button[aria-label="Record"] span,
.mma-card .gradio-audio button[aria-label="Record"] div,
.mma-card .gradio-audio button[aria-label*="Record"] span,
.mma-card .gradio-audio button[aria-label*="Record"] div,
.mma-card .gradio-audio [data-testid="record"] span,
.mma-card .gradio-audio [data-testid="record"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}


/* -------------------- AUDIO -------------------- */

.mma-card .gradio-audio,
.mma-card .gradio-audio > div,
.mma-card .gradio-audio .wrap,
.mma-card .gradio-audio .container {
    width: 100% !important;
    min-width: 0 !important;
    background: #ffffff !important;
    color: #172033 !important;
    color-scheme: light !important;
}

.mma-card .gradio-audio button {
    background: #ffffff !important;
    color: #344054 !important;
    -webkit-text-fill-color: #344054 !important;
    border: 1px solid #cfd8e6 !important;
    border-radius: 9px !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

.mma-card .gradio-audio button:hover {
    background: #f3f6fb !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
}

.mma-card .gradio-audio button svg {
    color: inherit !important;
    fill: currentColor !important;
}


/* -------------------- UPLOADS / IMAGE -------------------- */

.mma-card .upload-container,
.mma-card .dropzone,
.mma-card .file-preview,
.mma-card .gradio-image > div,
.mma-card .gradio-audio > div {
    background: #f8fafc !important;
    border-color: #dbe3ef !important;
    border-radius: 12px !important;
    color: #344054 !important;
}

.mma-card .upload-container *,
.mma-card .dropzone *,
.mma-card .file-preview * {
    color: #344054 !important;
    -webkit-text-fill-color: #344054 !important;
    overflow-wrap: anywhere !important;
}


/* -------------------- OUTPUTS -------------------- */

.mma-output {
    width: 100% !important;
    min-width: 0 !important;
    background: #f8fafc !important;
    border: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.mma-output textarea {
    background: #f8fafc !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #e1e7ef !important;
}


/* -------------------- SOURCES -------------------- */

.mma-source-box {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 16px !important;
    background: #f8fafc !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 13px !important;
    overflow: hidden !important;
    color: #000000 !important;
}

.mma-source-box * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.mma-source-title {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    margin: 0 0 11px !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 13px !important;
    line-height: 19px !important;
    font-weight: 750 !important;
}

.mma-source-title .mma-icon {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.mma-source {
    width: 100% !important;
    max-width: none !important;
    padding: 12px 0 0 !important;
    margin: 12px 0 0 !important;
    border-top: 1px solid #dbe3ef !important;
    overflow: hidden !important;
}

.mma-source-name {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 12px !important;
    line-height: 17px !important;
    font-weight: 700 !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

.mma-source-page {
    margin-top: 4px !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 11px !important;
    line-height: 16px !important;
}

.mma-source-preview {
    margin-top: 6px !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 11px !important;
    line-height: 17px !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}


/* -------------------- DISCLAIMER -------------------- */

.mma-disclaimer {
    display: flex !important;
    align-items: flex-start !important;
    gap: 9px !important;
    margin-top: 18px !important;
    padding: 12px 14px !important;
    background: #f8fafc !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 11px !important;
    color: #667085 !important;
    font-size: 11px !important;
    line-height: 17px !important;
}

.mma-disclaimer .mma-icon {
    font-size: 0 !important;
    width: 18px !important;
}

.mma-disclaimer .mma-icon::before {
    content: "ⓘ";
    font-size: 16px;
    color: #667085;
}

.mma-disclaimer span:last-child {
    color: #667085 !important;
}


/* -------------------- FOOTER -------------------- */

.mma-footer {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 24px !important;
    width: 100% !important;
    margin-top: 26px !important;
    padding: 16px 20px !important;
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 15px !important;
    color: #667085 !important;
    font-size: 11px !important;
    line-height: 17px !important;
}

.mma-footer strong {
    color: #172033 !important;
    font-size: 12px !important;
}


/* -------------------- LAYOUT -------------------- */

.mma-shell .gr-row:has(.mma-card) {
    align-items: flex-start !important;
}


/* -------------------- RESPONSIVE -------------------- */

@media (max-width: 900px) {

    .mma-shell {
        width: min(100% - 24px, 760px) !important;
        padding-top: 18px !important;
    }

    .mma-header-content {
        align-items: flex-start !important;
        flex-direction: column !important;
    }

    .mma-security {
        white-space: normal !important;
    }
}


@media (max-width: 700px) {

    .mma-shell {
        width: calc(100% - 20px) !important;
        padding: 12px 0 25px !important;
    }

    .mma-header {
        padding: 19px !important;
    }

    .mma-brand h1 {
        font-size: 23px !important;
        line-height: 29px !important;
    }

    .mma-brand p {
        font-size: 9px !important;
    }

    .mma-intro h2 {
        font-size: 23px !important;
    }

    .mma-tabs .tab-nav {
        flex-direction: column !important;
        height: auto !important;
    }

    .mma-tabs .tab-nav button,
    .mma-tabs .tab-nav button[role="tab"] {
        width: 100% !important;
        justify-content: center !important;
    }

    .mma-footer {
        align-items: flex-start !important;
        flex-direction: column !important;
    }
}
"""


# ============================================================
# RAG BACKEND
# ============================================================

@lru_cache(maxsize=1)
def load_rag_chain():

    rag_groq_api_key = os.getenv("RAG_GROQ_API_KEY")

    if not rag_groq_api_key:
        raise RuntimeError(
            "RAG_GROQ_API_KEY was not found. "
            "Please add it to the Render environment variables."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=512,
        api_key=rag_groq_api_key,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful medical knowledge assistant.

Your job is to answer general medical questions using ONLY
the information contained in the provided context.

Do not invent medical facts that are not supported by the context.

If the answer cannot be found in the provided documents, say:

"Because this information is not in the medical knowledge base,
the system cannot provide a reliable answer."

Keep the response clear and understandable.

This is an educational medical information system.

Do not present the response as a diagnosis or personal medical advice.

Context:

{context}

Question:

{input}
"""
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    doc_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    return create_retrieval_chain(
        retriever,
        doc_chain
    )


# ============================================================
# GENERAL MEDICAL CHATBOT
# ============================================================

def answer_medical_question(question, history):

    if not question or not question.strip():
        return history, ""

    question = question.strip()

    try:

        rag_chain = load_rag_chain()

        response = rag_chain.invoke(
            {
                "input": question
            }
        )

        answer = response["answer"]

        context_documents = response.get(
            "context",
            []
        )

        # ----------------------------------------------------
        # BUILD SOURCE DISPLAY
        # ----------------------------------------------------

        if context_documents:

            source_html = """
            <div class="mma-source-box">

                <div class="mma-source-title">

                    <span class="mma-icon">▤</span>

                    Retrieved Medical Sources

                </div>
            """

            for i, doc in enumerate(
                context_documents,
                1
            ):

                source = doc.metadata.get(
                    "source",
                    "Unknown source"
                )

                page = doc.metadata.get(
                    "page",
                    "N/A"
                )

                preview = doc.page_content[:400]

                safe_source = html.escape(
                    str(source)
                )

                safe_page = html.escape(
                    str(page)
                )

                safe_preview = html.escape(
                    str(preview)
                )

                source_html += f"""
                <div class="mma-source">

                    <div class="mma-source-name">

                        Source {i}: {safe_source}

                    </div>

                    <div class="mma-source-page">

                        Page: {safe_page}

                    </div>

                    <div class="mma-source-preview">

                        {safe_preview}

                    </div>

                </div>
                """

            source_html += """
            </div>
            """

        else:

            source_html = """
            <div class="mma-source-box">

                <div class="mma-source-title">

                    <span class="mma-icon">▤</span>

                    No supporting sources

                </div>

                <div style="
                    color:#667085 !important;
                    -webkit-text-fill-color:#667085 !important;
                    font-size:12px;
                    line-height:18px;
                ">

                    No supporting documents were retrieved
                    for this question.

                </div>

            </div>
            """

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        history = history or []

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return history, source_html

    except Exception as e:

        print(
            "RAG ERROR:",
            repr(e)
        )

        error_message = (
            "I couldn't process your question right now. "
            "Please check the RAG configuration and try again."
        )

        history = history or []

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": error_message
            }
        )

        return history, ""


# ============================================================
# MULTIMODAL MEDICAL ASSISTANT
# ============================================================

def process_inputs(
    audio_filepath,
    image_filepath
):

    if not audio_filepath:

        raise gr.Error(
            "Please record or upload your voice description first."
        )

    if not image_filepath:

        raise gr.Error(
            "Please upload a medical image before analysis."
        )

    try:

        # ----------------------------------------------------
        # PATIENT VOICE -> TEXT
        # ----------------------------------------------------

        patient_text = transcribe_patient_voice(
            audio_filepath
        )

        # ----------------------------------------------------
        # IMAGE + TEXT -> AI RESPONSE
        # ----------------------------------------------------

        doctor_text = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=image_filepath,
        )

        # ----------------------------------------------------
        # AI RESPONSE -> AUDIO
        # ----------------------------------------------------

        doctor_audio = convert_text_to_doctor_audio(
            doctor_text
        )

        return (
            patient_text,
            doctor_text,
            str(Path(doctor_audio)),
        )

    except Exception as e:

        print(
            "MULTIMODAL ERROR:",
            repr(e)
        )

        raise gr.Error(
            "Something went wrong while processing your "
            "medical concern. Please check your inputs "
            "and try again."
        )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title=APP_TITLE
) as iface:

    with gr.Column(
        elem_classes="mma-shell"
    ):

        # ====================================================
        # HEADER
        # ====================================================

        gr.HTML(
            """
            <header class="mma-header">

                <div class="mma-header-content">

                    <div class="mma-brand">

                        <div class="mma-brand-icon">
                            medical_services
                        </div>

                        <div>

                            <h1>
                                Multimodal Medical Assistant
                            </h1>

                            <p>
                                AI-POWERED MEDICAL INFORMATION &
                                MULTIMODAL ASSISTANCE
                            </p>

                        </div>

                    </div>


                    <div class="mma-security">

                        <span class="mma-icon">
                            ✓
                        </span>

                        <span style="color:#000000;">

                            Educational &amp;
                            privacy-conscious
                            AI assistance

                        </span>

                    </div>

                </div>

            </header>
            """
        )


        # ====================================================
        # INTRO
        # ====================================================

        gr.HTML(
            """
            <div class="mma-intro">

                <h2>
                    How can we help?
                </h2>

                <p>

                    Choose between general medical knowledge
                    grounded in a curated medical encyclopedia,
                    or a multimodal assistant that can work
                    with medical images and voice descriptions.

                </p>

            </div>
            """
        )


        # ====================================================
        # TABS
        # ====================================================

        with gr.Tabs(
            elem_classes="mma-tabs"
        ):

            # ==================================================
            # TAB 1 — GENERAL MEDICAL CHAT
            # ==================================================

            with gr.Tab(
                "📚 General Medical Chat"
            ):

                gr.HTML(
                    """
                    <div class="mma-mode-info">

                        <span class="mma-icon">
                            ▤
                        </span>

                        <div>

                            <strong>
                                Medical Knowledge Chatbot
                            </strong>

                            <span>

                                Ask general medical questions
                                and receive answers grounded
                                in the medical encyclopedia.

                            </span>

                        </div>

                    </div>
                    """
                )


                with gr.Row(
                    equal_height=False
                ):

                    # ------------------------------------------
                    # CHAT SECTION
                    # ------------------------------------------

                    with gr.Column(
                        scale=8
                    ):

                        with gr.Column(
                            elem_classes="mma-card"
                        ):

                            gr.HTML(
                                """
                                <div class="mma-section-title">

                                    <span class="mma-icon">
                                        💬
                                    </span>

                                    <h2>
                                        Ask a Medical Question
                                    </h2>

                                </div>
                                """
                            )


                            chatbot = gr.Chatbot(
                                label="Conversation",
                                elem_classes="mma-chatbot",
                                height=450,
                            )


                            with gr.Row(
                                equal_height=True
                            ):

                                medical_question = gr.Textbox(

                                    placeholder=(
                                        "Ask something about a disease, "
                                        "symptom, condition, treatment, "
                                        "or medical concept..."
                                    ),

                                    label="Your Question",

                                    lines=3,

                                    scale=8,

                                    elem_classes="mma-chat-input",

                                )


                                ask_button = gr.Button(
                                    "Ask",
                                    variant="primary",
                                    scale=2,
                                    min_width=100,
                                )


                            clear_button = gr.Button(
                                "Clear Conversation",
                                variant="secondary",
                            )


                    # ------------------------------------------
                    # SOURCES SECTION
                    # ------------------------------------------

                    with gr.Column(
                        scale=4
                    ):

                        with gr.Column(
                            elem_classes="mma-card"
                        ):

                            gr.HTML(
                                """
                                <div class="mma-section-title">

                                    <span class="mma-icon">
                                        ◈
                                    </span>

                                    <h2>
                                        Knowledge Sources
                                    </h2>

                                </div>
                                """
                            )


                            sources_output = gr.HTML(
                                """
                                <div class="mma-source-box">

                                    <div class="mma-source-title">

                                        <span class="mma-icon">
                                            ▤
                                        </span>

                                        Sources will appear here

                                    </div>

                                    <div style="
                                        color:#667085;
                                        font-size:12px;
                                        line-height:18px;
                                    ">

                                        After you ask a question,
                                        the documents retrieved
                                        by the RAG system will be
                                        displayed here.

                                    </div>

                                </div>
                                """
                            )


                            gr.HTML(
                                """
                                <div class="mma-disclaimer">

                                    <span class="mma-icon">
                                        ⓘ
                                    </span>

                                    <span>

                                        This chatbot provides
                                        educational medical
                                        information based on
                                        its knowledge base.
                                        It is not a substitute
                                        for professional medical
                                        advice, diagnosis, or
                                        treatment.

                                    </span>

                                </div>
                                """
                            )


            # ==================================================
            # TAB 2 — MULTIMODAL ASSISTANT
            # ==================================================

            with gr.Tab(
                "🩺 Multimodal Medical Assistant"
            ):

                gr.HTML(
                    """
                    <div class="mma-mode-info">

                        <span class="mma-icon">
                            ✚
                        </span>

                        <div>

                            <strong>
                                Multimodal Medical Assistance
                            </strong>

                            <span>

                                Provide a medical image and
                                describe your concern using
                                your voice. The system returns
                                AI-generated informational
                                guidance in text and audio.

                            </span>

                        </div>

                    </div>
                    """
                )


                with gr.Row(
                    equal_height=False
                ):

                    # ------------------------------------------
                    # PATIENT INPUT
                    # ------------------------------------------

                    with gr.Column(
                        scale=5
                    ):

                        gr.HTML(
                            """
                            <div class="mma-section-title">

                                <span class="mma-icon">
                                    ▤
                                </span>

                                <h2>
                                    Patient Input
                                </h2>

                            </div>
                            """
                        )


                        with gr.Column(
                            elem_classes="mma-card"
                        ):

                            gr.HTML(
                                """
                                <div style="
                                    color:#667085;
                                    font-size:12px;
                                    line-height:18px;
                                    margin-bottom:16px;
                                ">

                                    Provide both an image and
                                    a voice description for
                                    the multimodal analysis.

                                </div>
                                """
                            )


                            audio_input = gr.Audio(
                                sources=[
                                    "microphone",
                                    "upload"
                                ],
                                type="filepath",
                                label="Voice Description",
                            )


                            image_input = gr.Image(
                                type="filepath",
                                label="Medical Image",
                                height=300,
                            )


                            with gr.Column(
                                elem_classes="mma-primary-button"
                            ):

                                analyze_button = gr.Button(
                                    "Analyze Concern",
                                    variant="primary",
                                    size="lg",
                                )


                            gr.HTML(
                                """
                                <div class="mma-disclaimer">

                                    <span class="mma-icon">
                                        ⓘ
                                    </span>

                                    <span>

                                        Upload a clear image
                                        and describe the concern
                                        accurately. AI output
                                        is informational and
                                        should not be treated
                                        as a medical diagnosis.

                                    </span>

                                </div>
                                """
                            )


                    # ------------------------------------------
                    # AI RESPONSE
                    # ------------------------------------------

                    with gr.Column(
                        scale=7
                    ):

                        gr.HTML(
                            """
                            <div class="mma-section-title">

                                <span class="mma-icon">
                                    ✦
                                </span>

                                <h2>
                                    AI Response
                                </h2>

                            </div>
                            """
                        )


                        with gr.Column(
                            elem_classes="mma-card"
                        ):

                            transcript_output = gr.Textbox(
                                label="Your Speech Transcript",
                                lines=4,
                                interactive=False,
                                elem_classes="mma-output",
                            )


                            response_output = gr.Textbox(
                                label="AI Medical Guidance",
                                lines=10,
                                interactive=False,
                                elem_classes="mma-output",
                            )


                            audio_output = gr.Audio(
                                label="Audio Response",
                                type="filepath",
                                autoplay=True,
                            )


        # ====================================================
        # FOOTER
        # ====================================================

        gr.HTML(
            """
            <footer class="mma-footer">

                <div>

                    <strong>
                        Multimodal Medical Assistant
                    </strong>

                    <br/>

                    AI-powered educational healthcare
                    assistance.

                </div>


                <div>

                    AI-generated information is not a
                    medical diagnosis. Consult a qualified
                    healthcare professional when needed.

                </div>

            </footer>
            """
        )


    # ========================================================
    # EVENTS
    # ========================================================

    ask_button.click(
        fn=answer_medical_question,
        inputs=[
            medical_question,
            chatbot
        ],
        outputs=[
            chatbot,
            sources_output
        ],
    )


    medical_question.submit(
        fn=answer_medical_question,
        inputs=[
            medical_question,
            chatbot
        ],
        outputs=[
            chatbot,
            sources_output
        ],
    )


    clear_button.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[
            chatbot,
            sources_output
        ],
    )


    analyze_button.click(
        fn=process_inputs,
        inputs=[
            audio_input,
            image_input
        ],
        outputs=[
            transcript_output,
            response_output,
            audio_output
        ],
    )


# ============================================================
# LAUNCH — RENDER
# ============================================================

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    print(f"Starting Gradio on port {port}...")

    iface.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False,
        css=CSS,
        theme=gr.themes.Soft(),
        prevent_thread_lock=True,
    )

    print("Loading RAG system...")
    load_rag_chain()
    print("RAG system loaded successfully.")

    # Keep the process alive
    import time
    while True:
        time.sleep(60)
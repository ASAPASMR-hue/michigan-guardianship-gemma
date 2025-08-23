# System Overview: Michigan Guardianship AI

This document provides a high-level summary of the Michigan Guardianship AI project, outlining its goals, architecture, and key components.

## 1. Core Project Goal & Philosophy

*   **Primary Goal**: To be a production-ready, empathetic AI assistant that helps users navigate the complex process of minor guardianship in Michigan, with a laser focus on **Genesee County**.
*   **Core Philosophy**: The system prioritizes **"actionability"** and **"zero hallucination."** The goal isn't just to provide legally accurate text, but to give users clear, step-by-step guidance they can act upon. It is designed to be a safe and reliable resource for families.

## 2. System Architecture: The RAG Pipeline

This is a sophisticated Retrieval-Augmented Generation (RAG) system. The end-to-end process for handling a user query is as follows:

1.  **Query Intake**: A user submits a question through the Flask-based web UI.
2.  **Query Classification**: The system first classifies the query's complexity ("simple," "standard," or "complex"). This is a key step for the adaptive nature of the pipeline.
3.  **Adaptive Retrieval**: Based on the complexity, the system dynamically adjusts its search parameters (e.g., how many documents to retrieve, `top_k`).
4.  **Hybrid Search**: It performs a hybrid search to find relevant information in the knowledge base:
    *   **Vector Search**: For finding semantically similar content.
    *   **Lexical Search (BM25)**: For matching specific keywords and legal terms.
    *   A mandatory filter is applied to ensure results are specific to "Genesee County."
5.  **Reranking**: The retrieved document chunks are then re-ranked using a specialized reranker model (`BAAI/bge-reranker-v2-m3`) to push the most relevant information to the top.
6.  **Prompt Construction**: The top-ranked, reranked chunks are compiled into a comprehensive prompt for the Large Language Model (LLM). This prompt includes a strict legal disclaimer, instructions on the "zero hallucination" policy, and guidance on switching between a "strict" legal mode and a "personalized" empathetic mode.
7.  **LLM Generation**: An LLM (the system is designed to be compatible with models like Mixtral or Qwen) generates the user-facing answer based on the rich prompt.
8.  **Post-Generation Validation**: Before the response is sent to the user, it goes through a crucial validation pipeline that checks for:
    *   **Hallucinations**: Using the `lettuce-detect` library.
    *   **Citation Compliance**: Ensuring every legal fact is properly cited.
    *   **Procedural Accuracy**: Verifying that Genesee County-specific details (like the $175 filing fee or Thursday hearing dates) are correct.
    *   **Scope**: Confirming the query is within the defined scope of minor guardianship.
9.  **Deliver Response**: The final, validated, and formatted response is delivered to the user in the chat interface.

## 3. Knowledge Base (`kb_files/`)

*   The "brain" of the system is the `kb_files/` directory.
*   It contains a curated set of documents, including:
    *   Numbered knowledge base articles explaining different aspects of guardianship law.
    *   Official court forms (PDFs and text).
    *   Instructive documents that guide the AI's behavior.

## 4. Application & Interface

*   The main application is run via `app.py`, which starts a Flask web server.
*   The user interface is a simple web-based chat application, with files located in the `ui/` and `guardianship_ui/` directories.
*   The server manages conversation history and state through a session-based system.

## 5. Configuration & Setup

*   **Environment**: API keys (Google AI, HuggingFace) and server port settings are managed through a `.env` file.
*   **Dependencies**: All Python dependencies are listed in `requirements.txt`.
*   **Automated Setup**: The `jules-setup/` directory contains shell scripts (`jules_setup.sh`, `jules_quick_setup.sh`) that completely automate the environment setup, including creating a virtual environment and installing all required packages. This makes onboarding very straightforward.

# Document Q&A Chatbot

A Streamlit application for asking questions about PDF, CSV, and XLSX documents, with a document-free chat mode.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   `pip install streamlit huggingface_hub PyPDF2 openpyxl`
3. Copy `.env.example` to `.env` and set `HF_TOKEN` if your Hugging Face model access requires authentication.
4. Run the app:

   `streamlit run app.py`

## Notes

- The app keeps the recent conversation in Streamlit session state.
- Document text is currently truncated to 6,000 characters before being sent to the model. Large-document retrieval and citations are not implemented yet.
- Do not commit `.env` or real access tokens.

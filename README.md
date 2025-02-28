FetchFacts 
A Streamlit web app that fetches articles from user-provided URLs, extracts meaningful content, and answers user queries using FAISS for semantic search and FLAN-T5 for text generation.

Features
- Fetches articles from given URLs.
- Splits text into manageable chunks for processing.
- Embeds documents using sentence-transformers/all-mpnet-base-v2.
- Uses FAISS for fast document retrieval.
- Generates answers using Google's flan-t5-small model.

Usage Guide
- Enter up to 3 article URLs in the sidebar.
- Click "Process URLs" to fetch and analyze the content.
- Enter a question in the input field.
- Get AI-generated answers with references to the source documents.
import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from transformers import pipeline

st.title("FetchFacts")
st.sidebar.title("Article URL")

# User can enter the URL of the article
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Process the URL
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

# Load Hugging Face LLM
llm_pipeline = pipeline("text-generation", model="google/flan-t5-small")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Reading the articles")
    data = loader.load()

    if data:
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Splitting the text in the articles")
        docs = text_splitter.split_documents(data)

        if docs:
            # Create embeddings and save to FAISS index
            main_placeholder.text("Embedding the text in the articles")
            doc_texts = [doc.page_content for doc in docs]
            doc_embeddings = embedding_model.embed_documents(doc_texts)  
            vectorstore_hf = FAISS.from_embeddings(doc_embeddings, embedding_model, metadatas=[{"text": text} for text in doc_texts])  # ✅ Pass `embedding_model`
            
            # Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_hf, f)

            time.sleep(2)
        else:
            main_placeholder.text("Text Splitter produced empty documents. Check data.")
    else:
        main_placeholder.text("Data loading failed. Check URLs or network connection.")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)
            
            context = "\n\n".join([doc.page_content for doc in docs])  
            
            response = llm_pipeline(prompt, max_length=500, temperature=0.9)
            answer = response[0]["generated_text"]

            st.header("Answer")
            st.write(answer)

            # Display sources if available
            if docs:
                st.subheader("Sources:")
                for doc in docs:
                    st.write(doc.page_content)  
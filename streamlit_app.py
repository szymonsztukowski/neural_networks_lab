import streamlit as st
from openai import OpenAI
import os
import tempfile
from doc_loader import load_pdf
from embedder_rag import create_index, retrieve_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

# Sidebar for file upload
with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None and st.button("Process Document"):
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        try:
            text = load_pdf(tmp_file_path)
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            # Create LangChain documents
            documents = [Document(page_content=chunk, metadata={"filename": uploaded_file.name}) for chunk in chunks]
            # Create FAISS index
            faiss_index = create_index(documents)
            st.session_state["faiss_index"] = faiss_index
            st.success("Document processed and indexed!")
        finally:
            os.unlink(tmp_file_path)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    
    full_prompt = prompt
    # Retrieve relevant docs if index exists
    if "faiss_index" in st.session_state:
        relevant_docs = retrieve_docs(prompt, st.session_state["faiss_index"], k=3)
        context = "\n\n".join([doc["text"] for doc in relevant_docs])
        full_prompt = f"Context from documents:\n{context}\n\nQuestion: {prompt}"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(prompt)  # Display original prompt
    response = client.chat_completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
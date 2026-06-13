import streamlit as st
from openai import OpenAI
import os
import tempfile
from doc_loader import load_pdf
from embedder_rag import create_index, retrieve_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
    
    # Build client (new-style OpenAI client when available)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        client = None

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(prompt)  # Display original prompt

    response = None
    # Try new OpenAI client patterns first, then fall back to legacy `openai` package
    if client is not None:
        try:
            response = client.chat_completions.create(
                model=selected_model,
                messages=st.session_state.messages
            )
        except Exception:
            try:
                # alternative attribute name some versions expose
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=st.session_state.messages
                )
            except Exception:
                response = None

    if response is None:
        # Fallback to the older `openai` package API
        try:
            import openai as old_openai
            old_openai.api_key = api_key
            if base_url:
                old_openai.api_base = base_url
            response = old_openai.ChatCompletion.create(
                model=selected_model,
                messages=st.session_state.messages
            )
        except Exception as e:
            st.error(f"Failed to call chat completion API: {e}")
            st.stop()

    # Extract assistant message text robustly across client versions
    msg = None
    try:
        msg = response.choices[0].message.content
    except Exception:
        try:
            msg = response.choices[0].message["content"]
        except Exception:
            try:
                msg = response.choices[0].text
            except Exception:
                msg = str(response)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
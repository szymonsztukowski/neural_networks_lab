import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

# Sidebar for file upload
with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "py", "json"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    
    # Add file content if uploaded
    full_prompt = prompt
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        full_prompt += f"\n\nFile content ({uploaded_file.name}):\n{file_content}"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(prompt)  # Display original prompt, but send full
    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
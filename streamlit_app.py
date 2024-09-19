import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from collections import deque
import PyPDF2
import io
import requests
from bs4 import BeautifulSoup


st.title("HW 3 -- Enhanced Chatbot for URL Reading and Analysis")
st.write(
    "Upload a document, input URLs, and interact with various LLMs. "
    "Ask questions about the content and get answers from the selected AI model."
)


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return ""


if 'document_content' not in st.session_state:
    st.session_state.document_content = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=5)
if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = ""
if 'token_count' not in st.session_state:
    st.session_state.token_count = 0


st.sidebar.title("Options")


url1 = st.sidebar.text_input("Enter URL 1:")
url2 = st.sidebar.text_input("Enter URL 2:")


llm_vendor = st.sidebar.selectbox(
    "Choose LLM Vendor",
    ("OpenAI", "Claude", "Gemini")
)


if llm_vendor == "OpenAI":
    model = st.sidebar.selectbox("Choose OpenAI Model", ("gpt-3.5-turbo", "gpt-4o-mini"))
elif llm_vendor == "Claude":
    model = st.sidebar.selectbox("Choose Claude Model", ("claude-2", "claude-instant-1"))
else:
    model = "gemini-pro"


memory_type = st.sidebar.selectbox(
    "Choose Conversation Memory Type",
    ("Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens")
)


def get_llm_client():
    if llm_vendor == "OpenAI":
        return OpenAI(api_key=st.secrets["openai_api"])
    elif llm_vendor == "Claude":
        return Anthropic(api_key=st.secrets["claude_api"])
    else:
        genai.configure(api_key=st.secrets["gemini_api"])
        return genai.GenerativeModel(model)


def update_conversation_history(question, response):
    if memory_type == "Buffer of 5 questions":
        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
    elif memory_type == "Conversation summary":
        summary_prompt = f"Summarize the following conversation in 2-3 sentences:\nUser: {question}\nAssistant: {response}\nPrevious summary: {st.session_state.conversation_summary}"
        summary = generate_response(summary_prompt)
        st.session_state.conversation_summary = summary
    elif memory_type == "Buffer of 5,000 tokens":
        new_content = f"User: {question}\nAssistant: {response}\n"
        st.session_state.token_count += len(new_content.split())
        while st.session_state.token_count > 5000:
            removed = st.session_state.conversation_history.popleft()
            st.session_state.token_count -= len(removed['content'].split())
        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})


def generate_response(prompt):
    client = get_llm_client()
    
    if memory_type == "Buffer of 5 questions":
        conversation = list(st.session_state.conversation_history)
    elif memory_type == "Conversation summary":
        conversation = [{"role": "system", "content": f"Previous conversation summary: {st.session_state.conversation_summary}"}]
    else:  # Buffer of 5,000 tokens
        conversation = list(st.session_state.conversation_history)
    
    conversation.append({"role": "user", "content": prompt})
    
    if llm_vendor == "OpenAI":
        response = client.chat.completions.create(model=model, messages=conversation)
        return response.choices[0].message.content
    elif llm_vendor == "Claude":
        conversation_text = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        conversation_text += "\n\nAssistant:"
        response = client.completions.create(model=model, prompt=conversation_text, max_tokens_to_sample=1000)
        return response.completion
    else:
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        conversation_text += "\nAI:"
        response = client.generate_content(conversation_text)
        return response.text


uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))


if uploaded_file and not st.session_state.file_processed:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'txt':
        st.session_state.document_content = uploaded_file.read().decode()
    elif file_extension == 'pdf':
        st.session_state.document_content = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    else:
        st.error("Unsupported file type.")
        st.stop()
    st.session_state.file_processed = True


if not uploaded_file:
    st.session_state.document_content = None
    st.session_state.file_processed = False


url_content = ""
if url1:
    url_content += extract_text_from_url(url1) + "\n\n"
if url2:
    url_content += extract_text_from_url(url2)


combined_content = ""
if st.session_state.document_content:
    combined_content += st.session_state.document_content + "\n\n"
combined_content += url_content


question = st.text_area(
    "Now ask a question about the document or URLs!",
    placeholder="Can you give me a short summary?",
    disabled=not combined_content,
)

if combined_content and question:
    prompt = f"Here's the content: {combined_content} \n\n---\n\n {question}"
    with st.spinner("Generating response..."):
        response = generate_response(prompt)
        st.write("Assistant:", response)
        update_conversation_history(question, response)


st.write("## Conversation Memory")
if memory_type == "Conversation summary":
    st.write(f"Summary: {st.session_state.conversation_summary}")
else:
    for message in st.session_state.conversation_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
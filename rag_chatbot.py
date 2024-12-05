import streamlit as st
import os
from pypdf import PdfReader
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
from typing import List
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import tempfile

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Gemini API Key not found in environment variables")
    st.stop()

# Initialize session state
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'db' not in st.session_state:
    st.session_state.db = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# Configure page
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📚 Chat with your PDF using Text or Voice")

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        padding: 10px 20px;
        margin: 5px 0;
    }
    .voice-button {
        border-radius: 50% !important;
        height: 60px !important;
        width: 60px !important;
    }
    .stMarkdown {
        max-width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Voice processing function using Gemini
def process_audio_with_gemini(audio_bytes):
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for audio transcription
        prompt = "Please transcribe this audio accurately. Only return the transcribed text without any additional commentary."
        
        # Process audio with Gemini
        response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }
        ])
        
        # Extract transcribed text
        transcribed_text = response.text.strip()
        
        if transcribed_text:
            return transcribed_text
        else:
            st.warning("No speech detected. Please try speaking again.", icon="🎤")
            return None
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Existing PDF processing functions
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    return [i for i in re.split('\n\n', text) if i.strip()]

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Database management functions
def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    try:
        chroma_client.delete_collection(name=name)
    except:
        pass
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db

def get_relevant_passage(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'

ANSWER:
"""
    return prompt

def generate_answer(prompt: str):
    model = genai.GenerativeModel('gemini-1.5-flash')
    result = model.generate_content(prompt)
    return result.text

# Sidebar for PDF upload and management
with st.sidebar:
    st.header("PDF Management")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if st.session_state.pdf_name:
        st.info(f"Current PDF: {st.session_state.pdf_name}")
        if st.button("Remove Current PDF", key="remove_pdf", use_container_width=True):
            st.session_state.pdf_name = None
            st.session_state.db = None
            st.session_state.chat_history = []
            st.rerun()

# Process uploaded PDF
if uploaded_file is not None and (st.session_state.pdf_name != uploaded_file.name):
    with st.spinner("Processing PDF..."):
        db_folder = "chroma_db"
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunked_text = split_text(pdf_text)
        
        db_path = os.path.join(os.getcwd(), db_folder)
        db_name = "pdf_chat"
        st.session_state.db = create_chroma_db(chunked_text, db_path, db_name)
        st.session_state.pdf_name = uploaded_file.name
        st.rerun()

# Main chat interface
if st.session_state.pdf_name:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)
    
    # Create columns for text input and voice button
    col1, col2 = st.columns([0.9, 0.1])
    
    with col1:
        text_prompt = st.chat_input("Type your question or use the microphone →")
    
    with col2:
        st.markdown("<div style='display: flex; justify-content: center; padding-top: 10px;'>", unsafe_allow_html=True)
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            icon_size="2x",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            key="audio_recorder"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle voice input
    if audio_bytes and audio_bytes != st.session_state.audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        with st.spinner("Processing voice input..."):
            transcribed_text = process_audio_with_gemini(audio_bytes)
            if transcribed_text:
                # Process the transcribed text
                with st.chat_message("user"):
                    st.write(f"🎤 {transcribed_text}")
                
                st.session_state.chat_history.append({"role": "user", "content": f"🎤 {transcribed_text}"})
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        relevant_text = get_relevant_passage(transcribed_text, st.session_state.db, n_results=1)
                        final_prompt = make_rag_prompt(transcribed_text, "".join(relevant_text))
                        answer = generate_answer(final_prompt)
                        st.write(answer)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
    
    # Handle text input
    if text_prompt:
        with st.chat_message("user"):
            st.write(text_prompt)
        
        st.session_state.chat_history.append({"role": "user", "content": text_prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_text = get_relevant_passage(text_prompt, st.session_state.db, n_results=1)
                final_prompt = make_rag_prompt(text_prompt, "".join(relevant_text))
                answer = generate_answer(final_prompt)
                st.write(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.warning("Please upload a PDF to start chatting!")

# Instructions
st.markdown("---")
st.markdown("""
    ### 📝 Instructions
    1. Upload your PDF using the sidebar
    2. Choose how to ask questions:
        - Type your question in the text box
        - Click the microphone button and speak your question (wait for the red recording indicator)
    3. Wait for the AI to process and respond
    
    💡 **Tips for Voice Input**: 
    - Speak clearly and at a normal pace
    - Wait for the red recording indicator before speaking
    - Keep background noise to a minimum
    - Make sure your question is related to the uploaded PDF
""")
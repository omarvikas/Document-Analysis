import streamlit as st
import os
import tempfile
import openai
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# Page configuration
st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None
if "document_sentiment" not in st.session_state:
    st.session_state.document_sentiment = None
if "starter_questions" not in st.session_state:
    st.session_state.starter_questions = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

class SimpleRAGProcessor:
    """A simple RAG processor using OpenAI API directly and TF-IDF for document search"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_pdf(self, uploaded_file) -> str:
        """Load and extract text from PDF"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text from PDF
            text = ""
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def create_vectorstore(self, chunks: List[str]):
        """Create TF-IDF vectorstore for similarity search"""
        if not chunks:
            return None, None
            
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix
    
    def search_similar_chunks(self, query: str, vectorizer, tfidf_matrix, chunks: List[str], top_k: int = 3) -> List[str]:
        """Find most similar chunks to query"""
        if not vectorizer or tfidf_matrix is None:
            return []
            
        try:
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [chunks[i] for i in top_indices if similarities[i] > 0.1]
        except:
            return chunks[:top_k] if chunks else []
    
    def call_openai(self, prompt: str, max_tokens: int = 500) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def generate_summary(self, text: str) -> str:
        """Generate document summary"""
        # Truncate if too long
        if len(text) > 12000:
            text = text[:12000] + "..."
        
        prompt = f"""
        Please provide a comprehensive summary of the following document in exactly 250 words. 
        Focus on the main topics, key points, and important information:

        {text}

        Summary (250 words):
        """
        
        return self.call_openai(prompt, max_tokens=300)
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze document sentiment"""
        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "..."
        
        prompt = f"""
        Analyze the sentiment and tone of the following document. Provide:
        1. Overall sentiment (Positive/Negative/Neutral)
        2. Confidence level (High/Medium/Low)
        3. Key emotional indicators
        4. Brief explanation (2-3 sentences)

        Document text:
        {text}

        Sentiment Analysis:
        """
        
        return self.call_openai(prompt, max_tokens=200)
    
    def generate_starter_questions(self, text: str) -> List[str]:
        """Generate conversation starter questions"""
        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "..."
        
        prompt = f"""
        Based on the following document, generate exactly 2 thoughtful conversation starter questions 
        that would help someone explore the main topics and key insights. Make them specific and engaging.

        Document text:
        {text}

        Please provide exactly 2 questions, one per line, without numbering or bullets:
        """
        
        try:
            response = self.call_openai(prompt, max_tokens=150)
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            # Clean and take first 2
            cleaned_questions = []
            for q in questions[:2]:
                q = re.sub(r'^[0-9\.\-\*\•\s]+', '', q).strip()
                if q and q.endswith('?'):
                    cleaned_questions.append(q)
            
            if len(cleaned_questions) < 2:
                cleaned_questions.extend([
                    "What are the main topics discussed in this document?",
                    "What are the key insights from this document?"
                ])
                
            return cleaned_questions[:2]
        except:
            return [
                "What are the main topics discussed in this document?",
                "What are the key insights from this document?"
            ]
    
    def answer_question(self, question: str, relevant_chunks: List[str]) -> str:
        """Answer question based on relevant document chunks"""
        if not relevant_chunks:
            return "I can only answer questions based on the uploaded document. This information is not available in the current document."
        
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""
        You are a helpful assistant that answers questions based ONLY on the provided context from the uploaded document.

        IMPORTANT RULES:
        1. Only answer questions using information from the context below
        2. If the answer cannot be found in the context, politely decline and say "I can only answer questions based on the uploaded document. This information is not available in the current document."
        3. Be precise and cite specific parts of the document when possible
        4. Do not make up or infer information not explicitly stated in the context

        Context from document:
        {context}

        Question: {question}

        Answer:
        """
        
        return self.call_openai(prompt, max_tokens=400)

def main():
    st.title("📚 Document RAG Chatbot")
    st.markdown("Upload a PDF document and chat with it! Get summaries, sentiment analysis, and ask questions.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the chatbot"
        )
        
        st.markdown("---")
        
        # File upload
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file and openai_api_key:
            if st.button("Process Document", type="primary"):
                try:
                    with st.spinner("Processing document..."):
                        processor = SimpleRAGProcessor(openai_api_key)
                        
                        # Load PDF
                        document_text = processor.load_pdf(uploaded_file)
                        
                        if document_text:
                            # Create chunks
                            chunks = processor.chunk_text(document_text)
                            st.session_state.document_chunks = chunks
                            
                            # Create vectorstore
                            vectorizer, tfidf_matrix = processor.create_vectorstore(chunks)
                            st.session_state.vectorizer = vectorizer
                            st.session_state.tfidf_matrix = tfidf_matrix
                            
                            # Generate analysis
                            with st.spinner("Generating summary..."):
                                summary = processor.generate_summary(document_text)
                                st.session_state.document_summary = summary
                            
                            with st.spinner("Analyzing sentiment..."):
                                sentiment = processor.analyze_sentiment(document_text)
                                st.session_state.document_sentiment = sentiment
                            
                            with st.spinner("Generating starter questions..."):
                                questions = processor.generate_starter_questions(document_text)
                                st.session_state.starter_questions = questions
                            
                            st.session_state.document_processed = True
                            st.session_state.messages = []  # Clear previous chat
                            
                            st.success("Document processed successfully!")
                            st.rerun()
                        else:
                            st.error("Could not extract text from the PDF. Please try a different file.")
                            
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.error("Please check your OpenAI API key and try again.")
    
    # Main content area
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.info("You can get your API key from: https://platform.openai.com/api-keys")
        return
    
    if not st.session_state.document_processed:
        st.info("Please upload a PDF document in the sidebar to start chatting.")
        return
    
    # Document Analysis Section
    st.header("📊 Document Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.document_summary:
            st.subheader("📝 Summary")
            st.write(st.session_state.document_summary)
    
    with col2:
        if st.session_state.document_sentiment:
            st.subheader("😊 Sentiment Analysis")
            st.write(st.session_state.document_sentiment)
    
    # Starter Questions
    if st.session_state.starter_questions:
        st.subheader("💡 Suggested Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(st.session_state.starter_questions) > 0:
                if st.button(st.session_state.starter_questions[0], key="q1"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": st.session_state.starter_questions[0]
                    })
                    st.rerun()
        
        with col2:
            if len(st.session_state.starter_questions) > 1:
                if st.button(st.session_state.starter_questions[1], key="q2"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": st.session_state.starter_questions[1]
                    })
                    st.rerun()
    
    st.markdown("---")
    
    # Chat Interface
    st.header("💬 Chat with Document")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    processor = SimpleRAGProcessor(openai_api_key)
                    
                    # Find relevant chunks
                    relevant_chunks = processor.search_similar_chunks(
                        prompt, 
                        st.session_state.vectorizer,
                        st.session_state.tfidf_matrix,
                        st.session_state.document_chunks
                    )
                    
                    # Generate answer
                    response = processor.answer_question(prompt, relevant_chunks)
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
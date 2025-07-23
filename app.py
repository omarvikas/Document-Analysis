import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tempfile
import io
from typing import List, Optional

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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None
if "document_sentiment" not in st.session_state:
    st.session_state.document_sentiment = None
if "starter_questions" not in st.session_state:
    st.session_state.starter_questions = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

class DocumentProcessor:
    """Handles document processing and analysis"""
    
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
    def load_pdf(self, uploaded_file) -> List[Document]:
        """Load and process PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return documents
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return []
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        return vectorstore
    
    def generate_summary(self, documents: List[Document]) -> str:
        """Generate document summary"""
        # Combine all document content
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long (to avoid token limits)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        summary_prompt = f"""
        Please provide a comprehensive summary of the following document in exactly 250 words. 
        Focus on the main topics, key points, and important information:

        {full_text}

        Summary (250 words):
        """
        
        try:
            response = self.llm.predict(summary_prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_sentiment(self, documents: List[Document]) -> str:
        """Analyze document sentiment"""
        # Combine all document content
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long
        if len(full_text) > 6000:
            full_text = full_text[:6000] + "..."
        
        sentiment_prompt = f"""
        Analyze the sentiment and tone of the following document. Provide:
        1. Overall sentiment (Positive/Negative/Neutral)
        2. Confidence level (High/Medium/Low)
        3. Key emotional indicators
        4. Brief explanation (2-3 sentences)

        Document text:
        {full_text}

        Sentiment Analysis:
        """
        
        try:
            response = self.llm.predict(sentiment_prompt)
            return response.strip()
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"
    
    def generate_starter_questions(self, documents: List[Document]) -> List[str]:
        """Generate conversation starter questions"""
        # Combine all document content
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long
        if len(full_text) > 6000:
            full_text = full_text[:6000] + "..."
        
        questions_prompt = f"""
        Based on the following document, generate exactly 2 thoughtful conversation starter questions 
        that would help someone explore the main topics and key insights. Make them specific and engaging.

        Document text:
        {full_text}

        Please provide exactly 2 questions, one per line, without numbering:
        """
        
        try:
            response = self.llm.predict(questions_prompt)
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            # Take only first 2 questions and clean them
            questions = [q.lstrip('12.-• ').strip() for q in questions[:2]]
            return questions
        except Exception as e:
            return ["What are the main topics discussed in this document?", 
                   "What are the key insights from this document?"]

def create_qa_chain(vectorstore, openai_api_key: str):
    """Create QA chain with custom prompt"""
    
    # Custom prompt template
    prompt_template = """
    You are a helpful assistant that answers questions based ONLY on the provided context from the uploaded document.

    IMPORTANT RULES:
    1. Only answer questions using information from the context below
    2. If the answer cannot be found in the context, politely decline and say "I can only answer questions based on the uploaded document. This information is not available in the current document."
    3. Be precise and cite specific parts of the document when possible
    4. Do not make up or infer information not explicitly stated in the context

    Context: {context}

    Question: {question}

    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

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
                with st.spinner("Processing document..."):
                    processor = DocumentProcessor(openai_api_key)
                    
                    # Load PDF
                    documents = processor.load_pdf(uploaded_file)
                    
                    if documents:
                        # Create vectorstore
                        vectorstore = processor.create_vectorstore(documents)
                        st.session_state.vectorstore = vectorstore
                        
                        # Create QA chain
                        qa_chain = create_qa_chain(vectorstore, openai_api_key)
                        st.session_state.qa_chain = qa_chain
                        
                        # Generate analysis
                        with st.spinner("Generating summary..."):
                            summary = processor.generate_summary(documents)
                            st.session_state.document_summary = summary
                        
                        with st.spinner("Analyzing sentiment..."):
                            sentiment = processor.analyze_sentiment(documents)
                            st.session_state.document_sentiment = sentiment
                        
                        with st.spinner("Generating starter questions..."):
                            questions = processor.generate_starter_questions(documents)
                            st.session_state.starter_questions = questions
                        
                        st.session_state.document_processed = True
                        st.session_state.messages = []  # Clear previous chat
                        
                        st.success("Document processed successfully!")
                        st.rerun()
    
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
                    # Add question to chat
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": st.session_state.starter_questions[0]
                    })
                    st.rerun()
        
        with col2:
            if len(st.session_state.starter_questions) > 1:
                if st.button(st.session_state.starter_questions[1], key="q2"):
                    # Add question to chat
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
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    
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
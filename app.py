import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint # <-- CHANGE 1
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---- 1. SETUP: PDF Processing and RAG Pipeline ----

def load_and_process_pdf(pdf_path):
    """
    Loads a PDF, splits it into chunks, and creates a FAISS vector store.
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Use a cloud-hosted embedding model from Hugging Face
    # 'sentence-transformers/all-MiniLM-L6-v2' is a good, free option.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

def create_rag_chain(vectorstore, hf_token): # <-- CHANGE 2
    """
    Creates and returns the LangChain RAG pipeline using a Hugging Face model.
    """
    # Set the Hugging Face API token as an environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # Use a cloud-hosted LLM from Hugging Face
    # 'google/flan-t5-xxl' is a good, powerful model
    # Note: Many models on Hugging Face require a pro plan for their inference API.
    # Check the model page on the Hub for details.
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",
        task="text2text-generation",
        temperature=0.1
    )
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the prompt template
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question} 

    Context: {context} 

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Build the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ---- 2. STREAMLIT APPLICATION ----

st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📄")

def main():
    st.title("📄 PDF-based Q&A with RAG (Hugging Face)")
    st.markdown("This app uses Hugging Face models via their inference API. An API token is required.")

    # Get Hugging Face API token from Streamlit secrets
    hf_token = st.secrets.get("huggingface_api_token")
    if not hf_token:
        st.error("Hugging Face API token not found. Please add it to your Streamlit secrets.")
        st.stop()
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        
    if uploaded_file is not None:
        if st.session_state.vectorstore is None:
            # Save the uploaded file to a temporary location
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing PDF... This may take a moment."):
                try:
                    st.session_state.vectorstore = load_and_process_pdf("temp_pdf.pdf")
                    st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore, hf_token)
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    st.error("Please ensure your Hugging Face API token is correct and the models are available.")
                    
        st.success("PDF loaded successfully!")
        
        query = st.text_input("Ask a question about the document:", "")

        if query:
            if st.session_state.rag_chain:
                with st.spinner("Generating answer..."):
                    try:
                        response = st.session_state.rag_chain.invoke(query)
                        st.markdown(f"**Answer:** {response}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload a PDF first.")

if __name__ == "__main__":
    main()
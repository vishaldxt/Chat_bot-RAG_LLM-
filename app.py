import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

def create_rag_chain(vectorstore):
    """
    Creates and returns the LangChain RAG pipeline.
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125", 
        temperature=0, 
        openai_api_key=st.secrets["OPENAI_API_KEY"]
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
    st.title("📄 PDF-based Q&A with RAG")
    st.markdown("Upload a PDF and ask questions about its content.")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # --- Session State Management ---
    # We use session state to avoid reprocessing the PDF on every interaction.
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        
    if uploaded_file is not None:
        # Check if we need to re-process the PDF
        if st.session_state.vectorstore is None:
            # Save the uploaded file to a temporary location
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing PDF... This may take a moment."):
                st.session_state.vectorstore = load_and_process_pdf("temp_pdf.pdf")
                st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
                st.success("PDF processed successfully!")

        st.success("PDF loaded successfully!")
        
        # --- User Input & Chat Interface ---
        query = st.text_input("Ask a question about the document:", "")

        if query:
            if st.session_state.rag_chain:
                # Use a spinner while generating the response
                with st.spinner("Generating answer..."):
                    try:
                        response = st.session_state.rag_chain.invoke(query)
                        st.markdown(f"**Answer:** {response}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error("Please check your OpenAI API key and try again.")
            else:
                st.warning("Please upload a PDF first.")

if __name__ == "__main__":
    main()
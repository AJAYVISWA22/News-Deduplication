import streamlit as st
from RAG import ChatPDF  # Assuming RAG 2 is saved as rag2.py

# Initialize the ChatPDF instance
chat_pdf = ChatPDF()

# Sidebar for page navigation
st.sidebar.title("Pages")
pages = ['About Web App', 'PDF', 'URL']
selected_page = st.sidebar.selectbox("Navigate to:", pages)

# Page 1: About Web App
if selected_page == 'About Web App':
    st.title("About this Web App")
    st.write("""
        This web app allows you to interact with a powerful system that can process 
        both PDF documents and URLs to answer your questions based on the content 
        extracted from them. The system uses a Retrieval-Augmented Generation (RAG) model 
        with embeddings and a vector store for effective querying.
    """)
    st.write("Use the PDF or URL pages to upload and query documents.")

# Page 2: PDF Upload and Query
elif selected_page == 'PDF':
    st.title("PDF Document Upload")
    
    # Upload PDFs
    uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        pdf_file_paths = [file.name for file in uploaded_files]  # Storing PDF file names
        
        # Save the uploaded PDFs to the local system (for further processing)
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Ingest the PDFs
        chat_pdf.ingest(pdf_file_paths)
        st.success("PDFs have been ingested successfully!")

        # Ask a query related to the uploaded PDFs
        query = st.text_input("Ask a question based on the uploaded PDFs:")
        if query:
            response = chat_pdf.ask(query)
            st.write("Answer:", response)
    
    else:
        st.write("Please upload PDF files to begin processing.")

# Page 3: URL Input and Query
elif selected_page == 'URL':
    st.title("URL Input for Content Extraction")
    
    # Input for URLs
    urls_input = st.text_area("Enter URLs (separate with a newline):")
    if urls_input:
        urls = [url.strip() for url in urls_input.split("\n")]

        # Ingest the URLs
        chat_pdf.ingest_urls(urls)
        st.success("URLs have been ingested successfully!")

        # Ask a query related to the ingested URLs
        query = st.text_input("Ask a question based on the URLs:")
        if query:
            response = chat_pdf.ask(query)
            st.write("Answer:", response)

    else:
        st.write("Please enter URLs to begin processing.")


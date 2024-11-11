from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import filter_complex_metadata
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # Use a lighter, faster model for embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=50)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_paths: list):
        all_chunks = []
        with ThreadPoolExecutor() as executor:
            # Map each PDF processing to a thread and collect results
            results = executor.map(self.process_pdf, pdf_file_paths)
            for result in results:
                all_chunks.extend(result)

        # Create the FAISS vector store with batch processing for embeddings
        vector_store = FAISS.from_documents(documents=all_chunks, embedding=self.embedding_model)
        
       # Set up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3},
        )

        # Set up the chain
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def process_pdf(self, pdf_file_path):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        return filter_complex_metadata(chunks)

    def fetch_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()
            docs = [Document(page_content=text_content)]
            chunks = self.text_splitter.split_documents(docs)
            return filter_complex_metadata(chunks)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []

    def ingest_urls(self, urls):
        all_chunks = []
        for url in urls:
            chunks = self.fetch_url(url)
            all_chunks.extend(chunks)

        # Batch embeddings
        vector_store = FAISS.from_documents(documents=all_chunks, embedding=self.embedding_model)

        # Set up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3},
        )

        # Set up the chain
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add PDF documents or URLs first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None


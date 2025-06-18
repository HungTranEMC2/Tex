import time
import warnings

from datetime import datetime as dt

warnings.filterwarnings("ignore")

from langchain_communnity.document_loaders import PyPDFLoader

#Splitting data to managebale chunks and vectorizing in memory-store 
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndex

#Langchain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA 
from langchain.docstore.document import Document 

#HuggingFace Imports
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import faiss

class RAG:
    def __init__(self, pdf_path, model_name='Alibaba-NLP/gte-large-en-v1.5'):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents

    def create_vectorstore(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)

    def retriever(self, query, k=5):
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized, Run create_vectorstore first.")
        retriever = self.vectorstore.similarity_search(query, k=k)
        return retriever
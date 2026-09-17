from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

datapath="data/"

def load_pdf(data):
    
    loader=DirectoryLoader(data,glob='*pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    return document

document=load_pdf(data=datapath)
#print("Length of pages :", len(document))


def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=document)
#print("length of text chunks is ",len(text_chunks))

def load_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model=load_embedding_model()


db_faiss_path="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(db_faiss_path)

    
    


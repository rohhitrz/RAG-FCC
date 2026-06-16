import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import  (TextLoader, WebBaseLoader, PyPDFLoader, DirectoryLoader)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    # Create a temprory text file for demonstartion 

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b"this is a temprory text file.\nthis is a temprory text file")
        temp_file_path=temp_file.name

    try:
        # Load the text file using text loader
        loader=TextLoader(temp_file_path)
        documents=loader.load()

        # print(f"Loaded {len(documents)} document(s)")
        # print(f"Content preview: {documents[0].page_content[:100]}...")
        # print(f"Metadata: {documents[0].metadata}")


        # for doc in documents:
        #     print("Document content:")
        #     print(doc)
        #     print(doc.page_content)
    
    finally:
        os.remove(temp_file_path)


load_text_file()    



def doc_structure():
    pass


def pdf_Loader(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()


    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview:{doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")


pdf_Loader('./docs/langchain_demo.pdf')





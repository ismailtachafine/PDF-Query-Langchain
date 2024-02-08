# # LangChain components to use
# from langchain.vectorstores.cassandra import Cassandra
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
# # Support for dataset retrieval with Hugging Face
# from datasets import load_dataset
# # With CassIO, the engine powering the Astra DB integration in LangChain, you will also initialize the DB connection:
# import cassio
# # Read PDF
# from PyPDF2 import PdfReader

# # Environment variables
# import os
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# pdfreader = PdfReader('The Importance of Paragraph Writing.pdf')

# from typing_extensions import Concatenate
# raw_text = ''
# for i, page in enumerate(pdfreader.pages):
#     content = page.extract_text()
#     if content:
#         raw_text += content

# cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
# embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# astra_vector_store = Cassandra(
#     embedding=embedding,
#     table_name="LangchainVectorStore",
#     session=None,
#     keyspace=None,
# )

# from langchain.text_splitter import CharacterTextSplitter
# # We need to split the text using Character Text Split such that it sshould not increse token size
# text_splitter = CharacterTextSplitter(
#     separator = "\n",
#     chunk_size = 800,
#     chunk_overlap  = 200,
#     length_function = len,
# )
# texts = text_splitter.split_text(raw_text)

# astra_vector_store.add_texts(texts[:50])
# astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# first_question = True
# while True:
#     if first_question:
#         query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
#     else:
#         query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

#     if query_text.lower() == "quit":
#         break

#     if query_text == "":
#         continue

#     first_question = False

#     print("\nQUESTION: \"%s\"" % query_text)
#     answer = astra_vector_index.query(query_text, llm=llm).strip()
#     print("ANSWER: \"%s\"\n" % answer)


from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import cassio
import os
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

pdf_path = st.file_uploader("Upload PDF file", type="pdf")

if pdf_path:
    pdfreader = PdfReader(pdf_path)

    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="LangchainVectorStore",
        session=None,
        keyspace=None,
    )

    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    astra_vector_store.add_texts(texts[:50])
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    question_counter = 0
    while True:
        question_counter += 1
        query_text = st.text_input(f"Question {question_counter} (Type 'quit' to exit): ", key=f"question_{question_counter}").strip()
        
        if query_text.lower() == "quit" or query_text == "":
            break

        st.write("QUESTION: \"%s\"" % query_text)
        answer = astra_vector_index.query(query_text, llm=llm).strip()
        st.write("ANSWER: \"%s\"\n" % answer)

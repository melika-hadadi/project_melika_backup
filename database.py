# src/database.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from pathlib import Path

# مسیر پوشه دیتابیس
DB_PATH = Path("data/chroma_db")

def get_vector_db():
    """اتصال به دیتابیس برداری"""
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=str(DB_PATH), embedding_function=embedding)

def add_documents(documents):
    """ذخیره اسناد جدید در دیتابیس"""
    db = get_vector_db()
    db.add_documents(documents)
    print(f"تعداد {len(documents)} سند با موفقیت ذخیره شد.")

def get_retriever():
    """برگرداندن ابزار جستجو برای موتور RAG"""
    db = get_vector_db()
    return db.as_retriever(search_kwargs={"k": 3})

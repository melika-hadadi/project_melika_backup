from scraper import scrape_poems
from rag_engine import vectorstore
from langchain_core.documents import Document
import os

def load_data_into_vectorstore():
    print("در حال جمع‌آوری اشعار از منابع...")
    poems = scrape_poems()

    if not poems:
        print("هیچ شعری جمع‌آوری نشد. ممکن است مشکلی در scraper.py وجود داشته باشد.")
        return

    print(f"تعداد {len(poems)} قطعه شعر جمع‌آوری شد.")

    docs = [
        Document(
            page_content=item["content"],
            metadata={
                "source": item.get("source", "نامشخص"),
                "title": item.get("title", "نامشخص"),
                "poet": item.get("poet", "نامشخص")
            }
        )
        for item in poems
    ]

    print("در حال اضافه کردن اسناد به VectorStore (ChromaDB)...")
    try:
        vectorstore.add_documents(docs)
        print(f"موفقیت: {len(docs)} شعر در دیتابیس ذخیره شد.")
        print(f"دیتابیس در مسیر '{os.path.abspath('./data')}' ذخیره شد.")
    except Exception as e:
        print(f"خطا در اضافه کردن اسناد به VectorStore: {e}")


if __name__ == "__main__":
    load_data_into_vectorstore()

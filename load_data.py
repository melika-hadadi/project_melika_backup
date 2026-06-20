import os
import shutil

from langchain_core.documents import Document

from scraper import scrape_poems
from rag_engine import build_vectorstore, add_documents_to_store


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def reset_db_folder():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_data_into_vectorstore():
    print("در حال استخراج شعرها از گنجور ...")
    poems = scrape_poems(per_poet_limit=30)

    print(f"تعداد شعرهای استخراج‌شده: {len(poems)}")

    if not poems:
        print("هیچ شعری پیدا نشد. اسکرپر را بررسی کن.")
        return

    docs = []
    for item in poems:
        content = item.get("content", "").strip()
        title = item.get("title", "").strip() or "شعر بی‌نام"
        poet = item.get("poet", "").strip() or "نامشخص"
        source = item.get("source", "").strip()

        if not content:
            continue

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "title": title,
                    "poet": poet,
                }
            )
        )

    print(f"تعداد اسناد آماده برای ذخیره: {len(docs)}")

    reset_db_folder()
    vectorstore = build_vectorstore()
    add_documents_to_store(vectorstore, docs)



    print(f"دیتابیس با موفقیت ساخته شد و {len(docs)} سند ذخیره شد.")


if __name__ == "__main__":
    load_data_into_vectorstore()

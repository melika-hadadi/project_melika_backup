import os
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# تنظیم مسیر دیتابیس
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "data")

print(f"Checking database in: {PERSIST_DIRECTORY}")

# بارگذاری مدل
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# بارگذاری دیتابیس
try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )

    # دریافت اطلاعات کلی
    print("\n--- اطلاعات دیتابیس ---")
    # گرفتن ۱۰ مورد اول برای بررسی
    docs = vectorstore.get(limit=10)
    
    if not docs or not docs['metadatas']:
        print("❌ خطای بحرانی: هیچ داده‌ای در دیتابیس پیدا نشد! احتمالا دیتابیس خالی است.")
    else:
        print(f"✅ تعداد کل تکه‌های ذخیره شده (تقریبی): {len(docs['ids'])}")
        print("\nنمونه‌ای از شاعران و عناوین موجود:")
        print("-" * 50)
        
        poets_found = set()
        for i in range(len(docs['metadatas'])):
            meta = docs['metadatas'][i]
            poet = meta.get('poet', 'نامشخص')
            title = meta.get('title', 'بدون عنوان')
            poets_found.add(poet)
            print(f"شاعر: {poet} | عنوان: {title}")
        
        print("-" * 50)
        print(f"لیست شاعران شناسایی شده در این ۱۰ مورد: {poets_found}")

except Exception as e:
    print(f"❌ خطا در هنگام خواندن دیتابیس: {e}")

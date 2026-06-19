import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import List, Dict, Any

# تنظیمات مسیر دیتابیس
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# بارگذاری مدل Embedding
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# بارگذاری دیتابیس Chroma
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=embeddings
)

def normalize(text: str) -> str:
    """پاکسازی ساده متن برای جستجو"""
    return text.strip().lower()

def search_poems_with_scores(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    جستجو در دیتابیس و بازگرداندن نتایج منحصربه‌فرد بر اساس محتوا.
    """
    # مرحله اول: گرفتن تعداد بیشتری نتیجه (مثلاً 20 تا) برای اینکه فضای کافی برای فیلتر کردن داشته باشیم
    # اگر فقط k=5 بگیریم و 4 تای اول تکراری باشند، فقط 1 نتیجه برمی‌گردد.
    search_k = k * 4 
    
    try:
        results = vectorstore.similarity_search_with_score(normalize(query), k=search_k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

    unique_results = []
    seen_contents = set() # برای جلوگیری از تکرار دقیق یک متن

    for doc, score in results:
        content = doc.page_content.strip()
        
        # اگر این تکه متن قبلاً در نتایج اضافه شده، آن را رد کن
        if content not in seen_contents:
            unique_results.append({
                "content": content,
                "title": doc.metadata.get("title", "بدون عنوان"),
                "poet": doc.metadata.get("poet", "نامشخص"),
                "source": doc.metadata.get("source", "گنجور"),
                "score": float(score)
            })
            seen_contents.add(content)
        
        # اگر به تعداد درخواستی (k) رسیدیم، متوقف شو
        if len(unique_results) >= k:
            break
            
    return unique_results
def answer_question(question: str) -> str:
    """
    دریافت سوال کاربر، پیدا کردن مرتبط‌ترین شعر و ارائه یک پاسخ ساده.
    """
    results = search_poems_with_scores(question, k=1)
    if not results:
        return "متأسفانه در دیتابیس شعری مرتبط با پرسش شما یافت نشد."
    
    best_poem = results[0]
    return f"مرتبط‌ترین بیت/قطعه:\n\n{best_poem['content']}\n\n(شاعر: {best_poem['poet']} - عنوان: {best_poem['title']})"



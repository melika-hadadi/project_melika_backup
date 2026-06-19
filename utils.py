# src/utils.py
import re

def clean_text(text: str) -> str:
    """پاک‌سازی فواصل اضافه و کاراکترهای نامطلوب"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def prepare_poem_chunk(title: str, poet: str, text: str) -> str:
    """فرمت‌بندی متن برای دیتابیس (۳ آرگومان)"""
    return f"عنوان: {title}\nشاعر: {poet}\nمتن:\n{text}"

import os
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document


# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


# -----------------------------
# Vector Store
# -----------------------------
def build_vectorstore():
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def add_documents_to_store(vectorstore, docs: List[Document]):
    if not docs:
        return
    vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except Exception:
        pass


# -----------------------------
# Text utilities
# -----------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def expand_query(query: str) -> str:
    """
    Query expansion using lightweight Persian poetic keywords.
    Keeps compatibility without requiring Ollama.
    """
    query = normalize(query)

    expansions = {
        "عشق": ["محبت", "دل", "یار", "معشوق", "محبوب", "وصال"],
        "غم": ["اندوه", "حزن", "فراق", "درد", "دلتنگی"],
        "فراق": ["جدایی", "هجران", "دوری", "بیداد"],
        "وصال": ["دیدار", "ملاقات", "وصل", "اتصال"],
        "شراب": ["می", "باده", "ساقی", "مستی"],
        "مرگ": ["فنا", "نیستی", "مرگ", "عالم باقی"],
        "خدا": ["حق", "الهی", "معبود", "پروردگار"],
        "بهار": ["سبزه", "گل", "طراوت", "نسیم"],
        "زندگی": ["عمر", "حیات", "روزگار", "دنیا"],
        "سفر": ["راه", "کوچ", "هجرت", "سیر"],
    }

    extra_terms = []
    for key, vals in expansions.items():
        if key in query:
            extra_terms.extend(vals)

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


def score_to_percent(raw_score: float, min_score: float, max_score: float) -> float:
    """
    Chroma distance -> similarity percent.
    Lower distance means better match.
    """
    if max_score == min_score:
        return 100.0
    normalized = 1.0 - ((raw_score - min_score) / (max_score - min_score))
    return max(0.0, min(100.0, normalized * 100.0))


def _highlight_text(text: str, query: str) -> str:
    if not text or not query:
        return text
    keywords = set(normalize(query).split())
    if not keywords:
        return text

    highlighted = text
    for kw in sorted(keywords, key=len, reverse=True):
        if len(kw) < 2:
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", highlighted)
    return highlighted


def _find_key_verses(content: str, query: str, max_items: int = 3) -> List[str]:
    if not content:
        return []

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []

    keywords = set(normalize(query).split())
    scored_lines: List[Tuple[int, str]] = []

    for line in lines:
        score = 0
        normalized_line = normalize(line)
        for kw in keywords:
            if kw and kw in normalized_line:
                score += 1
        if score > 0:
            scored_lines.append((score, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored_lines[:max_items]]


# -----------------------------
# Ollama integration
# -----------------------------
def _call_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=180,
            shell=False,
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip()
        else:
            err = (proc.stderr or "").strip()
            print(f"[Ollama stderr] {err}")
    except FileNotFoundError:
        print("Ollama not found in PATH.")
    except subprocess.TimeoutExpired:
        print("Ollama call timed out.")
    except Exception as e:
        print(f"Ollama Error: {e}")
    return None


def _build_ollama_prompt(question: str, result: Dict[str, Any]) -> str:
    content = result.get("content", "")
    title = result.get("title", "بدون عنوان")
    poet = result.get("poet", "نامشخص")
    source = result.get("source", "")

    return f"""
شما یک تحلیل‌گر حرفه‌ای شعر فارسی هستید.

وظیفه:
- شعر زیر را در ارتباط با پرسش کاربر تحلیل کن.
- تحلیل باید دقیق، روان، مختصر اما عمیق باشد.
- اگر مفاهیم عرفانی، عاشقانه، غم، فراق، وصال، شراب، خدا، مرگ یا زندگی در شعر دیده می‌شود، اشاره کن.
- در پایان یک جمع‌بندی کوتاه بده.

پرسش کاربر:
{question}

مشخصات شعر:
عنوان: {title}
شاعر: {poet}
منبع: {source}

متن شعر:
{content}
""".strip()


def _fallback_analysis(question: str, result: Dict[str, Any]) -> str:
    title = result.get("title", "بدون عنوان")
    poet = result.get("poet", "نامشخص")
    score = result.get("score", 0)
    key_verses = result.get("key_verses", [])

    out = [
        f"این نتیجه با پرسش شما ارتباط دارد و از نظر معنایی امتیاز {score:.2f}% گرفته است.",
        f"شعر «{title}» از {poet} انتخاب شده است.",
    ]

    if key_verses:
        out.append("بیت‌های مرتبط:")
        for v in key_verses:
            out.append(f"• {v}")

    out.append("تحلیل هوشمند کامل در دسترس نیست، اما این شعر از نظر واژگانی و معنایی نزدیک به موضوع شماست.")
    return "\n".join(out)


# -----------------------------
# Search
# -----------------------------
def search_poems_with_scores(query: str, k: int = 5) -> List[Dict[str, Any]]:
    vectorstore = build_vectorstore()
    expanded = expand_query(query)

    raw_results = vectorstore.similarity_search_with_score(normalize(expanded), k=max(k * 2, 10))
    if not raw_results:
        return []

    scores = [float(score) for _, score in raw_results]
    min_score = min(scores)
    max_score = max(scores)

    final_results: List[Dict[str, Any]] = []
    seen = set()

    for doc, raw_score in raw_results:
        title = doc.metadata.get("title", "بدون عنوان")
        poet = doc.metadata.get("poet", "نامشخص")
        source = doc.metadata.get("source", "")
        content = (doc.page_content or "").strip()

        dedup_key = f"{title}|{poet}|{content[:80]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        score = score_to_percent(float(raw_score), min_score, max_score)
        excerpt = content[:400] + ("..." if len(content) > 400 else "")
        highlighted_excerpt = _highlight_text(excerpt, query)
        key_verses = _find_key_verses(content, query, max_items=3)

        final_results.append(
            {
                "score": round(score, 2),
                "raw_score": float(raw_score),
                "title": title,
                "poet": poet,
                "source": source,
                "content": content,
                "excerpt": highlighted_excerpt,
                "key_verses": key_verses,
            }
        )

        if len(final_results) >= k:
            break

    return final_results


# -----------------------------
# Public analysis API
# -----------------------------
def answer_question(
    question: str,
    result: Optional[Dict[str, Any]] = None,
    use_ollama: bool = True,
) -> str:
    """
    If result is provided, analyze only that selected poem.
    Otherwise fallback to a generic answer based on search.
    """
    if result is None:
        return "برای تحلیل دقیق، ابتدا یک نتیجه را از جدول انتخاب کنید."

    if not result.get("key_verses"):
        result["key_verses"] = _find_key_verses(result.get("content", ""), question, max_items=3)

    if use_ollama:
        prompt = _build_ollama_prompt(question, result)
        response = _call_ollama(prompt)
        if response:
            return response

    return _fallback_analysis(question, result)

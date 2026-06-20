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

# اگر این مدل روی سیستم کند بود، می‌توانی به llama3.2:1b یا gemma2:2b تغییر بدهی.
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

OLLAMA_TIMEOUT_SECONDS = 60
MAX_OLLAMA_CONTENT_CHARS = 2500

_vectorstore_instance = None


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """
    پاکسازی متن از نویزهای OCR و خروجی مدل.
    حذف الگوهایی مثل:
    [K], K], [K, 1D, 2D و مشابه آن‌ها
    """
    if not text:
        return ""

    text = str(text)

    # یکدست‌سازی حروف عربی/فارسی
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ۀ", "ه").replace("ة", "ه")

    # حذف کاراکترهای نامرئی
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", " ", text)

    # حذف نویزهای ترکیبی مثل [K] 1D
    text = re.sub(r"\[\s*[A-Za-z]\s*\]\s*\d+\s*[A-Za-z]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\s*[A-Za-z]?\s*\[\s*[A-Za-z]\s*\]", " ", text, flags=re.IGNORECASE)

    # حالت‌هایی که نویز وسط دو حرف فارسی چسبیده
    text = re.sub(r"([آ-ی])\s*\[\s*[A-Za-z]\s*\]\s*([آ-ی])", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([آ-ی])\s*\[?\s*[A-Za-z]\s*\]?\s*([آ-ی])", r"\1\2", text, flags=re.IGNORECASE)

    # حذف حالت‌های باقی‌مانده
    text = re.sub(r"\[\s*[A-Za-z]\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z]\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*[A-Za-z]\b", " ", text, flags=re.IGNORECASE)

    # حذف حروف لاتین تکی
    text = re.sub(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", " ", text)

    # حذف کدهای OCR مثل 1D، 2D، 10A
    text = re.sub(r"(?<![A-Za-z0-9])\d+\s*[A-Za-z](?![A-Za-z0-9])", " ", text, flags=re.IGNORECASE)

    # براکت خالی
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"\(\s*\)", " ", text)

    # اصلاح چند واژه احتمالی بعد از پاکسازی
    common_fixes = {
        "شع ر": "شعر",
        "مع نا": "معنا",
        "عش ق": "عشق",
        "فر اق": "فراق",
        "وص ال": "وصال",
        "خ دا": "خدا",
        "انس ان": "انسان",
        "ز ندگی": "زندگی",
        "مر گ": "مرگ",
        "تح لیل": "تحلیل",
        "بیت ها": "بیت‌ها",
        "واژه ها": "واژه‌ها",
        "نشانه ها": "نشانه‌ها",
    }

    for wrong, right in common_fixes.items():
        text = text.replace(wrong, right)

    # مرتب‌سازی فاصله‌ها
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def final_clean_response(text: str) -> str:
    """
    پاکسازی نهایی پاسخ مدل قبل از نمایش.
    """
    if not text:
        return ""

    text = str(text)

    for _ in range(3):
        text = clean_text(text)
        text = re.sub(r"\[\s*[A-Za-z]\s*\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[A-Za-z]\s*\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*[A-Za-z]\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z0-9])\d+\s*[A-Za-z](?![A-Za-z0-9])", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def normalize(text: str) -> str:
    if not text:
        return ""

    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _truncate_for_ollama(text: str, max_chars: int = MAX_OLLAMA_CONTENT_CHARS) -> str:
    """
    برای جلوگیری از کندی Ollama، متن خیلی بلند را کوتاه می‌کند.
    """
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected_lines = []
    total = 0

    for line in lines:
        if total + len(line) > max_chars:
            break
        selected_lines.append(line)
        total += len(line) + 1

    if selected_lines:
        return "\n".join(selected_lines).strip()

    return text[:max_chars].strip()


# -----------------------------
# Vector Store
# -----------------------------
def build_vectorstore():
    """
    ساخت VectorStore فقط یک بار برای جلوگیری از کرش Chroma در Streamlit/Windows.
    """
    global _vectorstore_instance

    if _vectorstore_instance is not None:
        return _vectorstore_instance

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    _vectorstore_instance = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    return _vectorstore_instance


def add_documents_to_store(vectorstore, docs: List[Document]):
    if not docs:
        return

    cleaned_docs = []

    for doc in docs:
        cleaned_docs.append(
            Document(
                page_content=clean_text(doc.page_content),
                metadata={
                    k: clean_text(v) if isinstance(v, str) else v
                    for k, v in doc.metadata.items()
                },
            )
        )

    vectorstore.add_documents(cleaned_docs)

    try:
        vectorstore.persist()
    except Exception:
        pass


# -----------------------------
# Query Expansion
# -----------------------------
def expand_query(query: str) -> str:
    query = normalize(query)

    expansions = {
        "عشق": ["محبت", "دل", "یار", "معشوق", "محبوب", "وصال"],
        "معشوق": ["یار", "محبوب", "دلبر", "جانانه", "نگار"],
        "یار": ["معشوق", "محبوب", "دلبر", "جانانه"],
        "غم": ["اندوه", "حزن", "فراق", "درد", "دلتنگی"],
        "فراق": ["جدایی", "هجران", "دوری", "بیداد"],
        "جدایی": ["فراق", "هجران", "دوری"],
        "وصال": ["دیدار", "ملاقات", "وصل", "اتصال"],
        "شراب": ["می", "باده", "ساقی", "مستی"],
        "می": ["شراب", "باده", "ساقی", "مستی"],
        "مرگ": ["فنا", "نیستی", "عالم باقی"],
        "خدا": ["حق", "الهی", "معبود", "پروردگار"],
        "حق": ["خدا", "الهی", "پروردگار"],
        "بهار": ["سبزه", "گل", "طراوت", "نسیم"],
        "زندگی": ["عمر", "حیات", "روزگار", "دنیا"],
        "سفر": ["راه", "کوچ", "هجرت", "سیر"],
        "تنهایی": ["دوری", "فراق", "غربت", "بی‌کسی"],
        "دنیا": ["روزگار", "جهان", "گیتی", "فلک"],
        "زهد": ["توبه", "عبادت", "پارسایی", "واعظ"],
        "رندی": ["مستی", "آزادی", "خرابات", "میخانه"],
    }

    extra_terms = []
    for key, vals in expansions.items():
        if key in query:
            extra_terms.extend(vals)

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


# -----------------------------
# Scoring / Highlighting
# -----------------------------
def score_to_percent(raw_score: float, min_score: float, max_score: float) -> float:
    if max_score == min_score:
        return 100.0

    normalized_score = 1.0 - ((raw_score - min_score) / (max_score - min_score))
    percent = normalized_score * 100.0
    return max(0.0, min(100.0, percent))


def _highlight_text(text: str, query: str) -> str:
    if not text or not query:
        return clean_text(text)

    text = clean_text(text)
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
    content = clean_text(content)
    if not content:
        return []

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        parts = re.split(r"[؛.!؟?]", content)
        lines = [p.strip() for p in parts if p.strip()]

    keywords = set(normalize(expand_query(query)).split())
    scored_lines: List[Tuple[int, str]] = []

    for line in lines:
        score = 0
        normalized_line = normalize(line)

        for kw in keywords:
            if len(kw) >= 2 and kw in normalized_line:
                score += 1

        if score > 0:
            scored_lines.append((score, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)

    selected = [clean_text(line) for _, line in scored_lines[:max_items]]
    if not selected:
        selected = [clean_text(line) for line in lines[:max_items]]

    return selected


# -----------------------------
# Local Semantic Fallback Analysis
# -----------------------------
def _detect_themes(text: str, question: str = "") -> List[str]:
    combined = normalize(question + " " + text)

    theme_keywords = {
        "عشق و دلدادگی": [
            "عشق", "معشوق", "محبوب", "یار", "دلبر", "جانانه", "دل", "محبت", "عاشق"
        ],
        "غم و اندوه": [
            "غم", "اندوه", "حزن", "درد", "گریه", "اشک", "سوز", "سوخت", "بسوخت"
        ],
        "فراق و دوری": [
            "فراق", "جدایی", "هجران", "دوری", "بی‌کسی", "غربت", "هجرت"
        ],
        "وصال و دیدار": [
            "وصال", "وصل", "دیدار", "ملاقات", "رسیدن", "حضور"
        ],
        "عرفان و معنویت": [
            "خدا", "حق", "الهی", "پروردگار", "روح", "جان", "فنا", "بقا", "حقیقت"
        ],
        "رندی و می‌خواری نمادین": [
            "می", "شراب", "باده", "ساقی", "مستی", "میخانه", "خرابات", "رند"
        ],
        "ناپایداری دنیا": [
            "دنیا", "روزگار", "فلک", "زمانه", "عمر", "مرگ", "فنا", "نیستی"
        ],
        "امید و طراوت": [
            "بهار", "گل", "سبزه", "نسیم", "صبح", "روشن", "امید"
        ],
    }

    found = []
    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if kw in combined:
                found.append(theme)
                break

    return found


def _infer_tone(text: str, question: str = "") -> str:
    combined = normalize(question + " " + text)

    sad_words = ["غم", "درد", "فراق", "جدایی", "سوخت", "اشک", "گریه", "اندوه", "هجران"]
    mystical_words = ["حق", "خدا", "فنا", "بقا", "روح", "جان", "حقیقت", "الهی"]
    romantic_words = ["عشق", "یار", "معشوق", "محبوب", "دلبر", "جانانه"]
    hopeful_words = ["بهار", "صبح", "گل", "نسیم", "امید", "روشن"]

    if any(w in combined for w in sad_words):
        return "لحن شعر اندوهناک، سوزناک و آمیخته با حس فقدان یا دلتنگی است."
    if any(w in combined for w in mystical_words):
        return "لحن شعر تأملی و عرفانی است و به لایه‌ای فراتر از معنای ظاهری اشاره می‌کند."
    if any(w in combined for w in romantic_words):
        return "لحن شعر عاشقانه است و بر رابطه‌ی عاشق، معشوق و احوال دل تکیه دارد."
    if any(w in combined for w in hopeful_words):
        return "لحن شعر روشن، لطیف و امیدوارانه است."

    return "لحن شعر تأملی و شاعرانه است و بیشتر بر دریافت احساسی و معنایی تکیه دارد."


def _extract_keywords_for_analysis(text: str, question: str = "", max_items: int = 8) -> List[str]:
    combined = normalize(question + " " + text)

    candidates = [
        "عشق", "معشوق", "یار", "دل", "غم", "فراق", "وصال", "جان",
        "خدا", "حق", "دنیا", "مرگ", "زندگی", "می", "شراب", "ساقی",
        "بهار", "گل", "شب", "صبح", "آتش", "خانه", "کاشانه", "اشک",
        "درد", "امید", "تنهایی", "روزگار"
    ]

    found = []
    for word in candidates:
        if word in combined and word not in found:
            found.append(word)

    return found[:max_items]


def _fallback_analysis(question: str, result: Dict[str, Any]) -> str:
    """
    تحلیل جایگزین وقتی Ollama خاموش است یا پاسخ نمی‌دهد.
    این بخش دیگر فقط متن عمومی نمی‌دهد و بر اساس مضمون‌های شعر تحلیل می‌سازد.
    """
    title = clean_text(result.get("title", "بدون عنوان"))
    poet = clean_text(result.get("poet", "نامشخص"))
    score = result.get("score", 0)
    content = clean_text(result.get("content", ""))
    key_verses = result.get("key_verses", [])

    key_verses = [clean_text(v) for v in key_verses if clean_text(v)]
    themes = _detect_themes(content, question)
    tone = _infer_tone(content, question)
    keywords = _extract_keywords_for_analysis(content, question)

    if not key_verses:
        key_verses = _find_key_verses(content, question, max_items=3)

    out = []

    out.append(f"### تحلیل شعر «{title}» از {poet}")
    out.append("")
    out.append(f"این شعر از نظر معنایی با پرسش شما ارتباط دارد و امتیاز شباهت آن حدود {score:.2f}% است.")
    out.append("")

    if themes:
        out.append("**مضمون‌های اصلی:**")
        out.append("، ".join(themes))
        out.append("")

    out.append("**لحن و فضای شعر:**")
    out.append(tone)
    out.append("")

    if keywords:
        out.append("**واژه‌ها و نشانه‌های کلیدی:**")
        out.append("، ".join(keywords))
        out.append("")

    if key_verses:
        out.append("**بخش‌های مهم و مرتبط:**")
        for verse in key_verses:
            out.append(f"- {verse}")
        out.append("")

    out.append("**ارتباط با پرسش شما:**")
    out.append(_build_local_question_relation(question, themes, keywords))
    out.append("")

    out.append("**جمع‌بندی:**")
    out.append(_build_local_summary(themes, tone))

    out.append("")
    out.append("> نکته: این تحلیل با روش محلی ساخته شده است. اگر Ollama فعال باشد ولی پاسخ ندهد، احتمالاً مدل کند است یا هنوز کامل اجرا نشده.")

    return final_clean_response("\n".join(out))


def _build_local_question_relation(question: str, themes: List[str], keywords: List[str]) -> str:
    question = clean_text(question)

    if not question:
        return "این شعر از طریق مضمون‌ها و واژه‌های برجسته‌اش با فضای کلی جست‌وجوی شما مرتبط است."

    if themes:
        return (
            f"پرسش شما درباره‌ی «{question}» است. در این شعر، مضمون‌هایی مثل "
            f"{'، '.join(themes[:3])} دیده می‌شود و همین باعث می‌شود شعر از نظر معنایی به پرسش شما نزدیک باشد."
        )

    if keywords:
        return (
            f"پرسش شما درباره‌ی «{question}» است. حضور واژه‌هایی مانند "
            f"{'، '.join(keywords[:5])} باعث ارتباط این شعر با جست‌وجوی شما شده است."
        )

    return (
        f"پرسش شما درباره‌ی «{question}» است. ارتباط این شعر بیشتر از راه فضای کلی، لحن و نزدیکی معنایی با پرسش شما شکل گرفته است."
    )


def _build_local_summary(themes: List[str], tone: str) -> str:
    if themes:
        return (
            f"در مجموع، شعر بر محور {themes[0]} شکل گرفته و با زبانی شاعرانه، احساسی درونی را بیان می‌کند. "
            f"{tone}"
        )

    return (
        "در مجموع، شعر حال‌وهوایی تأملی دارد و بیشتر از راه تصویرسازی، لحن و واژه‌های کلیدی با مخاطب ارتباط برقرار می‌کند."
    )


# -----------------------------
# Ollama integration
# -----------------------------
def _call_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL) -> Optional[str]:
    """
    اجرای Ollama با timeout کوتاه‌تر.
    اگر مدل کند باشد یا خروجی ندهد، None برمی‌گرداند تا fallback اجرا شود.
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=OLLAMA_TIMEOUT_SECONDS,
            shell=False,
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode == 0 and stdout:
            return final_clean_response(stdout)

        if stderr:
            print(f"[Ollama stderr] {stderr}")
        else:
            print("[Ollama] خروجی خالی برگشت.")

    except FileNotFoundError:
        print("Ollama not found in PATH.")
    except subprocess.TimeoutExpired:
        print(f"Ollama call timed out after {OLLAMA_TIMEOUT_SECONDS} seconds.")
    except Exception as e:
        print(f"Ollama Error: {e}")

    return None


def _build_ollama_prompt(question: str, result: Dict[str, Any]) -> str:
    content = _truncate_for_ollama(result.get("content", ""))
    title = clean_text(result.get("title", "بدون عنوان"))
    poet = clean_text(result.get("poet", "نامشخص"))
    source = clean_text(result.get("source", ""))

    return f"""
شما یک تحلیل‌گر حرفه‌ای شعر فارسی هستید.

وظیفه:
شعر زیر را در ارتباط با پرسش کاربر تحلیل کن.

خروجی را دقیقاً با این ساختار بده:

### مضمون اصلی
در ۲ تا ۳ جمله توضیح بده شعر درباره چیست.

### لحن و فضا
لحن شعر را توضیح بده: عاشقانه، عارفانه، غمگین، فلسفی، امیدوارانه یا ترکیبی.

### واژه‌ها و تصویرهای کلیدی
مهم‌ترین واژه‌ها، تصویرها یا نمادهای شعر را توضیح بده.

### ارتباط با پرسش کاربر
بگو چرا این شعر به پرسش کاربر مربوط است.

### جمع‌بندی
یک جمع‌بندی کوتاه و روشن بده.

قواعد بسیار مهم:
- پاسخ فقط فارسی باشد.
- پاسخ کوتاه، دقیق و تحلیلی باشد.
- از کلی‌گویی زیاد پرهیز کن.
- هیچ نویز OCR یا متن نامفهوم ننویس.
- این موارد را هرگز در پاسخ ننویس:
  [K]
  K]
  [K
  1D
  2D
  3D
  و ترکیب‌های مشابه.
- اگر در شعر نشانه‌هایی از عشق، فراق، وصال، غم، عرفان، شراب، خدا، مرگ، زندگی یا ناپایداری دنیا هست، اشاره کن.
- اگر متن شعر کوتاه یا ناقص بود، با همان مقدار موجود تحلیل کن.

پرسش کاربر:
{clean_text(question)}

مشخصات شعر:
عنوان: {title}
شاعر: {poet}
منبع: {source}

متن شعر:
{content}
""".strip()


# -----------------------------
# Search
# -----------------------------
def search_poems_with_scores(query: str, k: int = 5) -> List[Dict[str, Any]]:
    vectorstore = build_vectorstore()
    expanded = expand_query(query)

    raw_results = vectorstore.similarity_search_with_score(
        normalize(expanded),
        k=max(k * 2, 10),
    )

    if not raw_results:
        return []

    scores = [float(score) for _, score in raw_results]
    min_score = min(scores)
    max_score = max(scores)

    final_results: List[Dict[str, Any]] = []
    seen = set()

    for doc, raw_score in raw_results:
        title = clean_text(doc.metadata.get("title", "بدون عنوان"))
        poet = clean_text(doc.metadata.get("poet", "نامشخص"))
        source = clean_text(doc.metadata.get("source", ""))
        content = clean_text((doc.page_content or "").strip())

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

    final_results = sorted(final_results, key=lambda x: x.get("score", 0), reverse=True)
    return final_results


# -----------------------------
# Public analysis API
# -----------------------------
def answer_question(
    question: str,
    result: Optional[Dict[str, Any]] = None,
    use_ollama: bool = True,
) -> str:
    if result is None:
        return "برای تحلیل دقیق، ابتدا یک نتیجه را از جدول انتخاب کنید."

    result = dict(result)
    result["title"] = clean_text(result.get("title", "بدون عنوان"))
    result["poet"] = clean_text(result.get("poet", "نامشخص"))
    result["source"] = clean_text(result.get("source", ""))
    result["content"] = clean_text(result.get("content", ""))

    if not result.get("key_verses"):
        result["key_verses"] = _find_key_verses(result.get("content", ""), question, max_items=3)
    else:
        result["key_verses"] = [
            clean_text(v)
            for v in result.get("key_verses", [])
            if clean_text(v)
        ]

    if use_ollama:
        prompt = _build_ollama_prompt(question, result)
        response = _call_ollama(prompt)

        if response:
            return final_clean_response(response)

        fallback = _fallback_analysis(question, result)
        return final_clean_response(
            fallback
            + "\n\n"
            + "⚠️ Ollama در زمان تعیین‌شده پاسخ نداد یا خروجی معتبر برنگرداند؛ بنابراین تحلیل جایگزین نمایش داده شد."
        )

    return _fallback_analysis(question, result)

import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


BASE_URL = "https://ganjoor.net"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


TARGET_PAGES = [
    {"poet": "حافظ", "url": "https://ganjoor.net/hafez"},
    {"poet": "سعدی", "url": "https://ganjoor.net/saadi"},
    {"poet": "مولانا", "url": "https://ganjoor.net/moulavi"},
    {"poet": "فردوسی", "url": "https://ganjoor.net/ferdousi"},
    {"poet": "خیام", "url": "https://ganjoor.net/khayyam"},
    {"poet": "عطار", "url": "https://ganjoor.net/attar"},
]


BAD_TITLES = {
    "",
    "گنجور",
    "Ganjoor",
    "ورود به گنجور",
    "ثبت نام",
    "ثبت‌نام",
    "پیشخوان",
    "نمایه",
    "جستجو",
    "حاشیه‌ها",
    "حاشیه ها",
}


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ۀ", "ه")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_soup(url: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"خطا در دریافت {url}: {e}")
        return None


def _clean_url(url: str) -> str:
    parsed = urlparse(url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return clean.rstrip("/")


def _is_ganjoor_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in {"ganjoor.net", "www.ganjoor.net"}


def _get_root_path(index_url: str) -> str:
    parsed = urlparse(index_url)
    return parsed.path.rstrip("/")


def _is_under_poet(url: str, root_path: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return path == root_path or path.startswith(root_path + "/")


def _is_poem_url(url: str, root_path: str) -> bool:
    """
    لینک شعر در گنجور معمولاً شامل shعدد است.
    """
    if not _is_ganjoor_url(url):
        return False

    if not _is_under_poet(url, root_path):
        return False

    path = urlparse(url).path.rstrip("/")
    parts = path.split("/")

    for part in parts:
        if re.fullmatch(r"sh\d+", part):
            return True

    return False


def _extract_links_from_page(url: str):
    soup = _get_soup(url)

    if not soup:
        return []

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not href:
            continue

        if href.startswith("#"):
            continue

        full_url = urljoin(BASE_URL, href)
        full_url = _clean_url(full_url)

        if not _is_ganjoor_url(full_url):
            continue

        links.append(full_url)

    return links


def _discover_poem_links(index_url: str, per_poet_limit: int = 30):
    """
    چند مرحله داخل لینک‌های شاعر می‌رود تا لینک‌های شعر را پیدا کند.
    """
    root_path = _get_root_path(index_url)

    to_visit = [_clean_url(index_url)]
    visited = set()
    poem_links = []
    poem_seen = set()

    max_pages_to_visit = 150

    while to_visit and len(visited) < max_pages_to_visit and len(poem_links) < per_poet_limit:
        current_url = to_visit.pop(0)
        current_url = _clean_url(current_url)

        if current_url in visited:
            continue

        visited.add(current_url)

        links = _extract_links_from_page(current_url)

        for link in links:
            if not _is_under_poet(link, root_path):
                continue

            if _is_poem_url(link, root_path):
                if link not in poem_seen:
                    poem_seen.add(link)
                    poem_links.append(link)

                    if len(poem_links) >= per_poet_limit:
                        break
            else:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

        time.sleep(0.15)

    return poem_links


def _is_bad_title(title: str) -> bool:
    title = _normalize_text(title)
    title = title.strip(" -|،:؛")

    if title in BAD_TITLES:
        return True

    bad_words = [
        "ورود به گنجور",
        "ثبت نام",
        "ثبت‌نام",
        "حساب کاربری",
        "نام کاربری",
        "گذرواژه",
        "رمز عبور",
    ]

    for word in bad_words:
        if word in title:
            return True

    return False


def _extract_title(soup: BeautifulSoup, content: str = "") -> str:
    """
    استخراج عنوان شعر.
    اگر عنوان صفحه خراب بود، از خط اول شعر استفاده می‌کند.
    """
    candidates = [
        soup.select_one("h1"),
        soup.select_one("h2"),
        soup.select_one("title"),
    ]

    for tag in candidates:
        if not tag:
            continue

        txt = _normalize_text(tag.get_text(" ", strip=True))

        txt = re.sub(r"\s*\|\s*گنجور.*$", "", txt)
        txt = re.sub(r"\s*-\s*گنجور.*$", "", txt)
        txt = txt.strip(" -|،:؛")

        if txt and not _is_bad_title(txt):
            return txt

    # اگر عنوان مناسب پیدا نشد، از خط اول شعر استفاده کن
    if content:
        lines = [x.strip() for x in content.splitlines() if x.strip()]

        if lines:
            first_line = _normalize_text(lines[0])
            if first_line and not _is_bad_title(first_line):
                return first_line[:80]

    return "شعر بی‌نام"


def _extract_poem_content(soup: BeautifulSoup) -> str:
    """
    استخراج متن شعر از صفحه گنجور
    """
    selectors = [
        "div.b",
        ".b",
        "div.poem",
        "div#poem",
        "article div.b",
        "div.content div.b",
    ]

    lines = []

    for selector in selectors:
        elems = soup.select(selector)

        if elems:
            for el in elems:
                txt = _normalize_text(el.get_text(" ", strip=True))

                if txt and len(txt) > 2:
                    lines.append(txt)

            if lines:
                break

    # fallback در صورتی که ساختار صفحه متفاوت باشد
    if not lines:
        possible_tags = soup.find_all(["p", "div"])

        for tag in possible_tags:
            txt = _normalize_text(tag.get_text(" ", strip=True))

            if len(txt) < 25:
                continue

            bad_words = [
                "گنجور",
                "حاشیه",
                "نظرات",
                "ارسال",
                "ورود",
                "عضویت",
                "جستجو",
                "نام کاربری",
                "گذرواژه",
            ]

            if any(bad in txt for bad in bad_words):
                continue

            lines.append(txt)

    deduped = []
    seen = set()

    for line in lines:
        line = _normalize_text(line)

        if not line:
            continue

        if line in seen:
            continue

        seen.add(line)
        deduped.append(line)

    return "\n".join(deduped).strip()


def scrape_poems(per_poet_limit: int = 30):
    """
    تابع اصلی جمع‌آوری اشعار
    """
    all_poems = []

    print("شروع استخراج اشعار از گنجور...")

    for item in TARGET_PAGES:
        poet = item["poet"]
        index_url = item["url"]

        print(f"\nجستجو برای اشعار {poet}...")

        poem_links = _discover_poem_links(
            index_url=index_url,
            per_poet_limit=per_poet_limit,
        )

        if not poem_links:
            print(f"❌ هیچ لینک شعری برای {poet} پیدا نشد.")
            continue

        print(f"✅ برای {poet} تعداد {len(poem_links)} لینک شعر پیدا شد.")

        count = 0

        for poem_url in poem_links:
            if count >= per_poet_limit:
                break

            soup = _get_soup(poem_url)

            if not soup:
                continue

            content = _extract_poem_content(soup)

            if len(content) < 20:
                continue

            title = _extract_title(soup, content=content)

            if not title:
                title = "شعر بی‌نام"

            poem = {
                "poet": poet,
                "title": title,
                "content": content,
                "source": poem_url,
            }

            all_poems.append(poem)
            count += 1

            print(f"  • {poet} | {title}")

            time.sleep(0.15)

    print(f"\nپایان استخراج. تعداد کل اشعار یافت شده: {len(all_poems)}")

    return all_poems


if __name__ == "__main__":
    poems = scrape_poems(per_poet_limit=10)
    print(f"تعداد نهایی: {len(poems)}")

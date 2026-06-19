import requests
from bs4 import BeautifulSoup

def scrape_poems():
    # لیست گسترده‌تری از آدرس‌ها برای تست و پر کردن دیتابیس
    urls = [
        # فردوسی
        {"url": "https://ganjoor.net/ferdousi/shahname/aghaz", "poet": "فردوسی", "title": "آغاز شاهنامه"},
        {"url": "https://ganjoor.net/ferdousi/shahname/sh1", "poet": "فردوسی", "title": "شاهنامه بخش ۱"},
        
        # حافظ
        {"url": "https://ganjoor.net/hafez/ghazal/sh1", "poet": "حافظ", "title": "غزل ۱"},
        {"url": "https://ganjoor.net/hafez/ghazal/sh2", "poet": "حافظ", "title": "غزل ۲"},
        {"url": "https://ganjoor.net/hafez/ghazal/sh3", "poet": "حافظ", "title": "غزل ۳"},
        {"url": "https://ganjoor.net/hafez/ghazal/sh4", "poet": "حافظ", "title": "غزل ۴"},
        {"url": "https://ganjoor.net/hafez/ghazal/sh5", "poet": "حافظ", "title": "غزل ۵"},
        
        # سعدی
        {"url": "https://ganjoor.net/saadi/ghazal/sh1", "poet": "سعدی", "title": "غزل ۱"},
        {"url": "https://ganjoor.net/saadi/ghazal/sh2", "poet": "سعدی", "title": "غزل ۲"},
        {"url": "https://ganjoor.net/saadi/ghazal/sh3", "poet": "سعدی", "title": "غزل ۳"},
        
        # مولانا
        {"url": "https://ganjoor.net/molana/ghazal/sh1", "poet": "مولانا", "title": "غزل ۱"},
        {"url": "https://ganjoor.net/molana/ghazal/sh2", "poet": "مولانا", "title": "غزل ۲"},
    ]

    poems = []
    print(f"در حال شروع اسکرپ از {len(urls)} منبع...")

    for item in urls:
        url = item["url"]
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # در سایت گنجور، متن شعر معمولاً در کلاس‌های خاصی قرار دارد
            # ما تمام divهایی که کلاس b دارند یا داخل بخش اصلی شعر هستند را می‌گیریم
            found_in_url = False
            for div in soup.find_all('div', class_='b'):
                text = div.get_text(strip=True)
                if len(text) > 20:
                    poems.append({
                        "content": text,
                        "source": url,
                        "title": item["title"],
                        "poet": item["poet"]
                    })
                    found_in_url = True
            
            if found_in_url:
                print(f"✅ موفق: {item['poet']} - {item['title']}")
            else:
                print(f"⚠️ هشدار: چیزی از {item['poet']} در آدرس {url} پیدا نشد.")

        except Exception as e:
            print(f"❌ خطا در {url}: {e}")

    return poems

if __name__ == "__main__":
    scraped_data = scrape_poems()
    print(f"\n--- نتیجه نهایی ---")
    print(f"تعداد کل قطعات استخراج شده: {len(scraped_data)}")

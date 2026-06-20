import streamlit as st
import pandas as pd

from rag_engine import search_poems_with_scores, answer_question

st.set_page_config(
    page_title="دستیار هوشمند اشعار",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Session State
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "selected_result_index" not in st.session_state:
    st.session_state.selected_result_index = 0
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "use_ollama" not in st.session_state:
    st.session_state.use_ollama = True


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ تنظیمات")
    st.session_state.theme = st.radio(
        "حالت نمایش",
        ["dark", "light"],
        index=0 if st.session_state.theme == "dark" else 1,
        format_func=lambda x: "تاریک" if x == "dark" else "روشن",
    )

    st.session_state.use_ollama = st.toggle(
        "فعال‌سازی تحلیل هوشمند Ollama",
        value=st.session_state.use_ollama,
    )

    st.markdown("---")
    st.markdown("## راهنما")
    st.info(
        """
        1. یک واژه یا مفهوم وارد کن.
        2. روی «جستجو» بزن.
        3. ابتدا جدول نتایج را ببین.
        4. یک نتیجه را برای تحلیل انتخاب کن.
        5. روی دکمه تحلیل هوشمند بزن.
        """
    )

    st.markdown("---")
    st.markdown("## نمونه جستجو")
    st.caption("عشق، فراق، شراب، مرگ، وصال، غم، خدا")

    if st.button("🧹 پاکسازی حافظه"):
        st.session_state.results = []
        st.session_state.query = ""
        st.session_state.selected_result_index = 0
        st.rerun()


# -----------------------------
# Styling
# -----------------------------
dark_mode = st.session_state.theme == "dark"

bg_color = "#0f172a" if dark_mode else "#ffffff"
text_color = "#f8fafc" if dark_mode else "#111827"
card_bg = "#1e293b" if dark_mode else "#f3f4f6"
border_color = "rgba(255,255,255,0.08)" if dark_mode else "#e5e7eb"
muted_color = "#94a3b8" if dark_mode else "#6b7280"

st.markdown(
    f"""
    <style>
    .stApp {{
        direction: rtl;
        text-align: right;
        background: {bg_color};
        color: {text_color};
    }}
    .main-title {{
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}
    .subtitle {{
        color: {muted_color};
        margin-bottom: 1rem;
    }}
    .poem-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        line-height: 2.0;
    }}
    .analysis-box {{
        background: #0f766e;
        color: white;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        line-height: 2.0;
    }}
    .small-muted {{
        color: {muted_color};
        font-size: 0.92rem;
    }}
    .score-pill {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #2563eb;
        color: white;
        font-weight: 700;
    }}
    a {{
        text-decoration: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">📚 دستیار هوشمند اشعار فارسی</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">جستجوی معنایی، نمایش نتایج به‌صورت جدول، و تحلیل هوشمند فقط برای نتیجه انتخاب‌شده</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Search Form
# -----------------------------
with st.form("search_form"):
    query = st.text_input(
        "جستجو در اشعار",
        value=st.session_state.query,
        placeholder="مثلاً: عشق، فراق، شراب، مرگ، وصال ...",
    )
    submitted = st.form_submit_button("🔎 جستجو")

if submitted:
    st.session_state.query = query.strip()
    if st.session_state.query:
        with st.spinner("در حال جستجو در پایگاه داده..."):
            st.session_state.results = search_poems_with_scores(st.session_state.query, k=5)
            st.session_state.selected_result_index = 0
    else:
        st.warning("لطفاً یک عبارت برای جستجو وارد کنید.")
        st.session_state.results = []

# -----------------------------
# Results
# -----------------------------
if st.session_state.results:
    results = st.session_state.results

    st.subheader("نتایج جستجو")

    table_data = []
    for i, r in enumerate(results):
        table_data.append(
            {
                "ردیف": i + 1,
                "امتیاز": f"{r['score']:.2f}%",
                "شاعر": r["poet"],
                "عنوان": r["title"],
                "لینک": r["source"],
                "بخشی از متن": r["excerpt"],
            }
        )

    df = pd.DataFrame(table_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "لینک": st.column_config.LinkColumn("لینک", display_text="مشاهده منبع"),
        },
    )

    st.markdown("---")

    options = [
        f"{i + 1}. {r['poet']} — {r['title']} | امتیاز: {r['score']:.2f}%"
        for i, r in enumerate(results)
    ]

    selected_label = st.selectbox(
        "یک نتیجه را برای تحلیل انتخاب کنید",
        options=options,
        index=min(int(st.session_state.selected_result_index), len(options) - 1),
    )

    selected_index = options.index(selected_label)
    st.session_state.selected_result_index = selected_index
    chosen = results[selected_index]

    st.markdown("### نتیجه انتخاب‌شده")
    st.markdown(
        f"""
        <div class="poem-card">
            <div><strong>شاعر:</strong> {chosen["poet"]}</div>
            <div><strong>عنوان:</strong> {chosen["title"]}</div>
            <div><strong>منبع:</strong> <a href="{chosen["source"]}" target="_blank">{chosen["source"]}</a></div>
            <div><strong>امتیاز شباهت:</strong> <span class="score-pill">{chosen["score"]:.2f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### متن شعر / بخش مرتبط")
    st.markdown(
        f"""
        <div class="poem-card" style="white-space: pre-wrap;">
            {chosen["excerpt"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if chosen.get("key_verses"):
        st.markdown("### بیت‌های مهم")
        for verse in chosen["key_verses"]:
            st.markdown(
                f"""
                <div class="poem-card" style="white-space: pre-wrap;">
                    {verse}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### میزان شباهت")
    st.progress(min(max(chosen["score"] / 100.0, 0.0), 1.0))

    if st.button("✨ تحلیل هوشمند این نتیجه"):
        with st.spinner("در حال تحلیل شعر انتخاب‌شده..."):
            analysis = answer_question(
                st.session_state.query,
                result=chosen,
                use_ollama=st.session_state.use_ollama,
            )

        st.markdown(
            f"""
            <div class="analysis-box">
                {analysis}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("برای شروع، یک عبارت جستجو وارد کن و دکمه جستجو را بزن.")

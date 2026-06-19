import streamlit as st
import pandas as pd
from rag_engine import search_poems_with_scores, answer_question

st.set_page_config(page_title="دستیار ادبی گنجور", layout="wide")
st.markdown("<style>.stApp { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)

st.title("📚 دستیار ادبی گنجور")

with st.form("search_form"):
    query = st.text_input("جستجو در اشعار:")
    submitted = st.form_submit_button("جستجو")

if submitted and query:
    with st.spinner("در حال جستجو و تحلیل..."):
        results = search_poems_with_scores(query, k=5)

        st.subheader("نتایج یافت‌شده")
        if results:
            df = pd.DataFrame(results)
            
            # تغییر نام ستون‌ها برای نمایش بهتر
            df_display = df[["score", "title", "poet", "source", "content"]].rename(
                columns={
                    "score": "امتیاز",
                    "title": "عنوان",
                    "poet": "شاعر",
                    "source": "منبع",
                    "content": "متن"
                }
            )
            
            # نمایش جدول با قابلیت کلیک روی لینک‌ها
            st.dataframe(
                df_display,
                use_container_width=True,
                column_config={
                    "منبع": st.column_config.LinkColumn("منبع", display_text="مشاهده در گنجور")
                }
            )
        else:
            st.write("نتیجه‌ای یافت نشد.")

        st.subheader("تحلیل هوشمند")
        st.write(answer_question(query))

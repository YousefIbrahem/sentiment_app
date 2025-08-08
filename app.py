import joblib
import re
import streamlit as st
# تحميل النموذج و TF-IDF
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# تنظيف النص (نفس دالة التدريب)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

# واجهة Streamlit
st.title("🔍 Twitter Sentiment Analyzer")
st.write("ادخل تغريدة أو جملة وسيتم تصنيف الشعور إلى: إيجابي، سلبي، أو محايد")
user_input = st.text_area("📝 أدخل النص هنا:")
if st.button("تحليل الشعور"):
    if user_input.strip() == "":
        st.warning("من فضلك أدخل نصًا أولًا.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == "positive":
            st.success("🎉 الشعور: إيجابي")
        elif prediction == "negative":
            st.error("😠 الشعور: سلبي")
        else:
            st.info("😐 الشعور: محايد")

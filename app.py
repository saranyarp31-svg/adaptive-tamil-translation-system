import streamlit as st
from transformers import pipeline
from langdetect import detect
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="Adaptive Tamil Translation System")

st.title("Adaptive Tamil Translation System")
st.write("Translate English / Other languages to Tamil")

@st.cache_resource
def load_translator():
    return pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="tam_Taml"
    )

translator = load_translator()

text = st.text_area("Enter text to translate")

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = translator(text, max_length=200)
        tamil_text = result[0]["translation_text"]

        st.subheader("Tamil Translation")
        st.success(tamil_text)

        # Text to Speech
        tts = gTTS(tamil_text, lang="ta")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name)


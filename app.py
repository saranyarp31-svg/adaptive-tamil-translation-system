import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from gtts import gTTS
import tempfile
import torch

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Adaptive Tamil Translation System",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Adaptive Tamil Translation System")
st.write("Translate **any language → Tamil** with text & voice output")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- LANGUAGE MAP ----------------
LANG_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn"
}

# ---------------- TRANSLATION FUNCTION ----------------
def translate_to_tamil(text):
    try:
        detected = detect(text)
    except:
        detected = "en"

    src_lang = LANG_MAP.get(detected, "eng_Latn")
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
        max_length=200
    )

    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    return detected, translated

# ---------------- UI ----------------
user_input = st.text_area(
    "✍️ Enter text in ANY language",
    height=150,
    placeholder="Type English / Hindi / French / Any language..."
)

if st.button("🔁 Translate to Tamil"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Translating..."):
            lang, tamil_text = translate_to_tamil(user_input)

        st.success("Translation Completed ✅")
        st.write(f"**Detected Language:** `{lang}`")
        st.text_area("📘 Tamil Translation", tamil_text, height=150)

        # ---------------- AUDIO ----------------
        tts = gTTS(text=tamil_text, lang="ta")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name)

        # ---------------- DOWNLOAD ----------------
        st.download_button(
            label="⬇️ Download Tamil Text",
            data=tamil_text,
            file_name="tamil_translation.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("🎓 M.Tech Project | Adaptive Tamil Translation System")

"""Web demo: type a Hinglish sentence and see the English translation and its sentiment.

Usage: streamlit run demo/app.py
"""

import html

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from hinglish_mt.clean import preprocess, restore, tokenize
from hinglish_mt.cmi import cmi
from hinglish_mt.decode import load_run, translate
from hinglish_mt.langid import EN, HI, tag
from hinglish_mt.sentiment import label_of

RUN = "run_f"
EXAMPLES = [
    "kal movie dekhne chalein? mujhe bahut excited feel ho raha hai",
    "yaar mera phone kharab ho gaya, ab kya karu",
    "@rahul bhai tum kab aa rahe ho? #party",
    "mujhe nahi pata tha ki wo itna accha gaata hai",
    "aaj office me bahut kaam tha, i am so tired",
    "ye movie bahut acchi thi, mujhe bahut pasand aayi",
    "service bahut kharab hai, mera paisa barbaad ho gaya",
]
COLOURS = {HI: "#ffd8a8", EN: "#a5d8ff"}
OTHER_COLOUR = "#e9ecef"


@st.cache_resource
def get_model():
    model, tokeniser, checkpoint = load_run(RUN)
    return model, tokeniser


@st.cache_resource
def get_sentiment_analyser():
    return SentimentIntensityAnalyzer()


def coloured_words(text):
    """HTML that shows each word on a background colour for its language."""
    tokens = tokenize(text)
    tags = tag(tokens)
    parts = []
    for token, language in zip(tokens, tags):
        colour = COLOURS.get(language, OTHER_COLOUR)
        style = f"background:{colour}; color:#212529; padding:2px 6px; margin:2px; border-radius:4px; display:inline-block"
        parts.append(f'<span style="{style}">{html.escape(token)}</span>')
    return " ".join(parts), tags


st.set_page_config(page_title="Hinglish to English")
st.title("Hinglish to English translation")
st.write("A GRU encoder-decoder with attention, trained from scratch on synthetic Hinglish and real tweets.")

example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
if example == "(type your own)":
    default_text = ""
else:
    default_text = example
text = st.text_area("Hinglish sentence", value=default_text, height=100)

if st.button("Translate", type="primary") and text.strip() != "":
    model, tokeniser = get_model()
    cleaned, originals = preprocess(text)
    output = translate(model, tokeniser, [cleaned], beam=4)[0]
    translation = restore(output, originals)

    st.subheader("Translation")
    st.success(translation)

    st.subheader("Sentiment of the translation")
    compound = get_sentiment_analyser().polarity_scores(translation)["compound"]
    st.info(f"{label_of(compound)} (VADER score {compound:+.2f}, from -1 to +1)")
    st.caption("An English sentiment tool reads our translation. It does not understand Hinglish itself.")

    st.subheader("Language of each input word")
    words_html, tags = coloured_words(cleaned)
    st.markdown(words_html, unsafe_allow_html=True)
    legend = (
        f'<span style="background:{COLOURS[HI]}; color:#212529; padding:2px 6px; border-radius:4px">Hindi</span> '
        f'<span style="background:{COLOURS[EN]}; color:#212529; padding:2px 6px; border-radius:4px">English</span> '
        f'<span style="background:{OTHER_COLOUR}; color:#212529; padding:2px 6px; border-radius:4px">Other</span>'
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(f"Code-Mixing Index: {cmi(tags):.1f} (0 = one language, 50 = evenly mixed)")

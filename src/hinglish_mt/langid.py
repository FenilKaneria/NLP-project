"""Rule-based token-level language tagger for Romanised Hinglish.

Three tags: English, Hindi, other. The base rule is the PHINC paper's: a token
is English if an English dictionary knows it, other if it carries no language
(placeholder, number, symbol), and Hindi otherwise. Two additions deal with the
places where that rule fails:

* HINDI_WORDS: Romanised Hindi words that an English frequency list also
  contains ("hai", "ke", "se"), because the list is built from web text.
* AMBIGUOUS: words that are valid in both languages ("to", "me", "is").
  They are tagged by the majority language of their neighbours, with a
  per-word default when the neighbours do not decide.

`tag(tokens)` is the only entry point callers use, so a learned tagger can
replace this one later.
"""

import re
from functools import lru_cache

from wordfreq import zipf_frequency

from hinglish_mt.clean import MASKS

EN = "en"
HI = "hi"
OTHER = "other"

# Zipf 3.0 is about one occurrence per million words.
ENGLISH_ZIPF_THRESHOLD = 3.0
CONTEXT_WINDOW = 2

HINDI_WORDS = frozenset(
    """
    aa aap aapka aapke aapki aapko ab abe abhi abi ache acha achha ae agar aj ak ap apna apne apni
    ata aur aye baba bach banda bane bas beech bhai bhi bi cha chai chi crore dal dar der desi di
    dia dil din dino du ek ga gale ge hae hai hain ham han har hara hath hay hee ho hoga hone hoo
    hu hua hui hum hume hun hy ik isi iss ja jab jai jake jana jane jb jeb ji jo ka kab kai kal
    kar kara karo kat kay kb ke ki ko koi kr ky kya kyu lag lakh lao le lena li liye lo mai maine
    mana mann mar mast mata mei mein mera mere meri mil mila milne mubarak na nahi nahin nai nam
    nay ne nh nhi ni pa pas pe pita pta ra rah raha rahe rahi rang ri ro sa saab sab salaam salam
    sar sara sari sb se si tab tak te tera tere teri tha thi tho ti tk toh tu tum umar un uss vo
    wah wala wale wali wo woh yaar ye yeh zara
    """.split()
)

# word -> tag used when the neighbours give no majority.
AMBIGUOUS = {
    "to": EN, "me": HI, "is": EN, "do": EN, "the": EN, "in": EN, "he": EN, "hi": HI,
    "main": HI, "de": HI, "man": EN, "us": EN, "are": EN, "so": EN, "or": EN, "re": HI,
    "bus": EN, "pass": EN, "sun": EN, "par": HI, "log": HI, "ha": HI, "ban": EN, "bar": EN,
    "bat": HI, "dam": HI, "pal": HI, "mile": HI, "teen": HI, "tune": HI, "mat": HI,
    "kam": HI, "char": HI, "loot": HI, "ma": HI, "ya": HI, "dum": HI, "bin": HI, "hue": HI,
    "maze": HI, "niche": HI, "sham": HI, "kia": HI, "chain": EN, "bore": EN, "door": EN,
    "beta": HI, "lane": EN, "pure": EN, "bare": HI, "sake": HI, "rat": HI, "inn": HI,
    "jung": HI, "gore": HI, "hen": HI,
}  # fmt: skip

SINGLE_LETTERS = {"i": EN, "a": EN, "u": EN, "r": EN, "y": EN, "n": EN, "h": HI, "k": HI}

# Carry no language: retweet marker, link fragments, HTML residue, laughter and fillers.
OTHER_WORDS = frozenset({"rt", "amp", "http", "https", "www", "com", "url", "lol", "lmao", "rofl", "omg", "bc"})
_LAUGH_RE = re.compile(r"(?:a*(?:h+a+){2,}h*|(?:h+e+){2,}h*|l+o+l+|h+m+|a+w+|u+m+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def squeezed_forms(word):
    """The word plus versions with stretched letters reduced ("jaldiii" -> "jaldii", "jaldi")."""
    forms = [word]
    for replacement in (r"\1\1", r"\1"):
        form = _REPEAT_RE.sub(replacement, word)
        if form not in forms:
            forms.append(form)
    return tuple(forms)


@lru_cache(maxsize=None)
def is_english_word(word):
    """True if the English dictionary knows the word or a de-stretched form of it."""
    for form in squeezed_forms(word):
        if zipf_frequency(form, "en") >= ENGLISH_ZIPF_THRESHOLD:
            return True
    return False


@lru_cache(maxsize=None)
def _lexical_tag(token):
    """Tag from the token alone. None means ambiguous: the neighbours decide."""
    if token in MASKS or not token[0].isalpha():
        return OTHER

    forms = squeezed_forms(token)
    for form in forms:
        if form in OTHER_WORDS or _LAUGH_RE.fullmatch(form):
            return OTHER

    shortest = forms[-1]
    if len(shortest) == 1:
        return SINGLE_LETTERS.get(shortest, OTHER)

    for form in forms:
        if form in AMBIGUOUS:
            return None
    for form in forms:
        if form in HINDI_WORDS:
            return HI

    if is_english_word(token):
        return EN
    return HI


def _ambiguous_default(token):
    for form in squeezed_forms(token):
        if form in AMBIGUOUS:
            return AMBIGUOUS[form]


def is_ambiguous(token):
    return _lexical_tag(token.lower()) is None


def tag(tokens):
    """Tag each token as "en", "hi" or "other"."""
    lowered = []
    lexical = []
    for token in tokens:
        token = token.lower()
        lowered.append(token)
        lexical.append(_lexical_tag(token))

    tags = []
    for i in range(len(lowered)):
        label = lexical[i]
        if label is None:
            before = lexical[max(0, i - CONTEXT_WINDOW) : i]
            after = lexical[i + 1 : i + 1 + CONTEXT_WINDOW]
            neighbours = before + after
            hindi = neighbours.count(HI)
            english = neighbours.count(EN)
            if hindi > english:
                label = HI
            elif english > hindi:
                label = EN
            else:
                label = _ambiguous_default(lowered[i])
        tags.append(label)
    return tags

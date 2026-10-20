"""Cleaning and masking for tweet-style text.

Mentions, hashtags and links are replaced by placeholders before translation
and put back afterwards, because they must stay unchanged.
"""

import re

import ftfy

USER = "<user>"
HASHTAG = "<hashtag>"
URL = "<url>"
MASKS = (USER, HASHTAG, URL)

# The ways a link is written in PHINC.
URL_FORMS = [
    r"https?\s?:\s?//(?:\S+|\s(?=\S*[./]\S)\S+)?",  # http://... and "http: //..."
    r"(?<![a-z])https?(?:/url\b|(?![a-z]))",  # "http/URL" or a cut-off "http"
    r"www\.\S+",
    r"pic\.twitter\.com/\S+",
    r"(?<![\w.])(?!\w*pic\.twitter\.com/)(?:[a-z0-9-]+\.)+(?:com|org|net|in)/\S+",  # twitter.com/...
]
URL_PATTERN = "|".join(URL_FORMS)

# A handle or hashtag stops where a link starts ("#taghttps://...").
NAME_PATTERN = r"(?:(?!https?://|pic\.twitter\.com/)\w)+"
USER_PATTERN = r"(?<!\w)@" + NAME_PATTERN
HASHTAG_PATTERN = r"(?<!\w)#" + NAME_PATTERN

# Links come first, so that '#' and '@' inside a link are not masked separately.
_MASK_RE = re.compile(
    f"(?P<url>{URL_PATTERN})|(?P<user>{USER_PATTERN})|(?P<hashtag>{HASHTAG_PATTERN})",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"<(user|hashtag|url)>")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_SPACE_RE = re.compile(r"\s+")
# Placeholders, words (with inner apostrophes), numbers, then any other single character.
_TOKEN_RE = re.compile(r"<(?:user|hashtag|url)>|[a-z]+(?:'[a-z]+)*|\d+(?:[.,:]\d+)*|\S", re.IGNORECASE)


def has_devanagari(text):
    return _DEVANAGARI_RE.search(text) is not None


def single_spaces(text):
    return _SPACE_RE.sub(" ", text).strip()


def fix_text(text):
    """Repair broken characters and collapse whitespace."""
    text = ftfy.fix_text(text)
    text = text.replace("�", "")
    return single_spaces(text)


def mask(text):
    """Replace mentions, hashtags and links with placeholders.

    Returns the masked text and a list of (kind, original text) in order of appearance.
    """
    originals = []
    pieces = []
    position = 0
    for match in _MASK_RE.finditer(text):
        kind = match.lastgroup
        originals.append((kind, match.group()))
        pieces.append(text[position : match.start()])
        pieces.append(f" <{kind}> ")
        position = match.end()
    pieces.append(text[position:])
    masked = "".join(pieces)
    return single_spaces(masked), originals


def restore(text, originals):
    """Put the originals back. Each placeholder takes the next original of its kind.

    A placeholder with no original left (the model produced too many) is removed.
    """
    waiting = {}
    for kind, original in originals:
        if kind not in waiting:
            waiting[kind] = []
        waiting[kind].append(original)

    pieces = []
    position = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        kind = match.group(1)
        pieces.append(text[position : match.start()])
        if kind in waiting and len(waiting[kind]) > 0:
            pieces.append(waiting[kind].pop(0))
        position = match.end()
    pieces.append(text[position:])
    restored = "".join(pieces)
    return single_spaces(restored)


def tokenize(text):
    """Split cleaned text into placeholders, words, numbers and single symbols."""
    return _TOKEN_RE.findall(text)


def preprocess(text):
    """Full cleaning for model input: fix encoding, mask, lowercase."""
    masked, originals = mask(fix_text(text))
    return masked.lower(), originals

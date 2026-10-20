"""Rule-based Devanagari to casual Roman ("Hinglish") transliteration.

The target is the spelling people type on social media: no accents, the
unwritten "a" dropped where speech drops it ("करना" -> "karna"), and the
common spellings for long vowels ("काम" -> "kaam", "कमरा" -> "kamra").

Steps for one word:
1. Split it into syllables (consonants, vowel, nasal sign).
2. Mark the unwritten "a" (schwa) that is not pronounced.
3. Spell each syllable, then write long vowels with one letter or two.

Known limits: step 2 ignores word structure ("अनुमति" gives "anumti"), and
English loanwords come out as they sound ("वोट" gives "wot").
"""

import re
import unicodedata
from dataclasses import dataclass

VIRAMA = "्"
NUKTA = "़"
ANUSVARA = "ं"
CHANDRABINDU = "ँ"
VISARGA = "ः"

# Long vowels are kept as capitals until the last step decides their spelling: (short, double).
LONG_VOWELS = {"A": ("a", "aa"), "I": ("i", "ee"), "U": ("u", "oo")}

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}  # fmt: skip
# Consonant with a dot below (nukta). "ड़" and "ढ़" are written "d" and "dh" in casual text.
NUKTA_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f"}
VOWELS = {
    "अ": "a", "आ": "A", "इ": "i", "ई": "I", "उ": "u", "ऊ": "U", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o",
}  # fmt: skip
# Vowel signs attached to a consonant.
MATRAS = {
    "ा": "A", "ि": "i", "ी": "I", "ु": "u", "ू": "U", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
}  # fmt: skip

# Frequent words whose usual spelling the rules do not produce.
EXCEPTION_SPELLINGS = {
    "वह": "wo", "यह": "ye", "वे": "wo", "नहीं": "nahi", "माँ": "maa", "मां": "maa",
    "वहाँ": "wahan", "वहां": "wahan", "वही": "wahi", "वजह": "wajah", "वक्त": "waqt",
    "वक़्त": "waqt", "वरना": "warna", "ॐ": "om", "हम": "hum",
    "ज़िंदगी": "zindagi", "जिंदगी": "zindagi", "ज़िन्दगी": "zindagi", "जिन्दगी": "zindagi",
}  # fmt: skip
# Unicode can store the same letter in two ways. The NFD form makes them equal.
EXCEPTIONS = {}
for hindi_word, roman_word in EXCEPTION_SPELLINGS.items():
    EXCEPTIONS[unicodedata.normalize("NFD", hindi_word)] = roman_word

LABIALS = ("p", "b", "m", "f")
ASPIRATED = ("kh", "gh", "ch", "jh", "th", "dh", "ph", "bh")

# Hindi full stops become ".", Devanagari digits become 0-9.
PUNCTUATION = {"।": ".", "॥": ".", "॰": "."}
for digit in range(10):
    PUNCTUATION[chr(0x0966 + digit)] = str(digit)
_PUNCTUATION = str.maketrans(PUNCTUATION)

_WORD_RE = re.compile(r"[ऀ-ॣॱ-ॿ]+")
_INVISIBLE_RE = re.compile(r"[​-‍﻿]")
# "ah" between two consonants.
_AH_RE = re.compile(r"(?<=[b-df-hj-np-tv-z])ah(?=[b-df-hj-np-tv-z])")


@dataclass
class Syllable:
    onset: list  # the consonants before the vowel
    vowel: str = None  # None: consonants with no vowel after them
    schwa: bool = False  # the vowel is the unwritten "a"
    independent: bool = False  # written as a full vowel letter, not as a sign on a consonant
    nasal: bool = False
    visarga: bool = False
    silent: bool = False  # the schwa is not pronounced

    @property
    def pronounced(self):
        return self.vowel is not None and not self.silent


def _parse(word):
    """Split a word into syllables."""
    syllables = []
    onset = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            i += 1
            if i < n and word[i] == NUKTA:
                roman = NUKTA_CONSONANTS.get(ch, CONSONANTS[ch])
                i += 1
            elif ch == "ज" and word[i : i + 2] == VIRAMA + "ञ":
                roman = "gy"  # gyan
                i += 2
            else:
                roman = CONSONANTS[ch]
            onset.append(roman)

            if i < n and word[i] == VIRAMA:
                # No vowel: the next consonant joins this one.
                i += 1
                continue
            if i < n and word[i] in MATRAS:
                syllable = Syllable(onset, MATRAS[word[i]])
                i += 1
            else:
                syllable = Syllable(onset, "a", schwa=True)
        elif ch in VOWELS:
            syllable = Syllable(onset, VOWELS[ch], independent=True)
            i += 1
        else:
            # A stray sign with nothing to attach to.
            i += 1
            continue

        onset = []
        while i < n and word[i] in (ANUSVARA, CHANDRABINDU, VISARGA):
            if word[i] == VISARGA:
                syllable.visarga = True
            else:
                syllable.nasal = True
            i += 1
        syllables.append(syllable)

    if len(onset) > 0:
        syllables.append(Syllable(onset))
    return syllables


def can_lose_schwa(syllable):
    return syllable.schwa and not syllable.nasal and not syllable.visarga


def _delete_schwas(syllables):
    """Mark the schwas that are not pronounced: at the end of the word and some in the middle."""
    last = syllables[-1]
    if can_lose_schwa(last) and len(syllables) > 1:
        after_cluster = len(last.onset) > 1 and last.onset[-1] in ("y", "r")  # satya, mitra
        after_i = last.onset == ["y"] and syllables[-2].vowel in ("i", "I")  # bhartiya
        after_gy = last.onset[-1] == "gy"
        if after_cluster or after_i or after_gy:
            last.silent = False
        else:
            last.silent = True

    # Right to left, so that "samajhna" keeps the schwa that "samjhana" would lose.
    for i in range(len(syllables) - 2, 0, -1):
        current = syllables[i]
        following = syllables[i + 1]
        single_consonants = len(current.onset) == 1 and len(following.onset) == 1
        if can_lose_schwa(current) and single_consonants and following.pronounced:
            current.silent = True


def _spell_onset(onset, vowel):
    parts = list(onset)
    for i, consonant in enumerate(parts):
        if consonant == "v" and vowel is not None:
            after_consonant = i > 0 and parts[i - 1] != "r"
            alone_before_open_vowel = len(parts) == 1 and vowel in ("A", "o", "au")
            if after_consonant or alone_before_open_vowel:
                parts[i] = "w"  # swagat, vishwas, wala; but vikas, vyakti, sarvajanik
        if i > 0 and consonant == parts[i - 1] and consonant in ASPIRATED:
            parts[i - 1] = consonant[0]  # accha, patthar
    return "".join(parts)


def _spell(syllables, final_nasal):
    """Spell the syllables. Long vowels stay as capitals A, I, U for the next step."""
    out = []
    for i, syllable in enumerate(syllables):
        is_last = i == len(syllables) - 1
        if not syllable.pronounced:
            out.append(_spell_onset(syllable.onset, None))
            continue
        out.append(_spell_onset(syllable.onset, syllable.vowel))

        vowel = syllable.vowel
        if syllable.independent and i > 0:
            previous = syllables[i - 1]
            if vowel == "e" and previous.vowel in ("a", "A", "i", "I"):
                vowel = "ye"  # gaye, liye, chahiye
            if vowel == "I" and previous.schwa:
                vowel = "yI"  # gayi, nayi; but bhai, padhai

        if syllable.nasal:
            if not is_last:
                next_onset = syllables[i + 1].onset
                if len(next_onset) > 0 and next_onset[0].startswith(LABIALS):
                    vowel += "m"
                else:
                    vowel += "n"
            elif syllable.schwa:
                vowel += "m"  # swayam, evam
            elif final_nasal:
                if vowel == "e" and not syllable.independent:
                    vowel += "in"  # mein, karein
                else:
                    vowel += "n"

        if syllable.visarga:
            vowel += "H"  # capital, so that the "ah" rule leaves it alone
        out.append(vowel)
    return "".join(out)


def _spell_long_vowels(text, nuclei, double_long):
    """Write each long vowel (A, I, U) with one letter or two. `nuclei` is the number of vowels."""
    out = []
    for i, ch in enumerate(text):
        if ch not in LONG_VOWELS:
            out.append(ch)
            continue
        short, double = LONG_VOWELS[ch]
        followed = i + 1 < len(text)
        only_vowel = nuclei == 1 and (followed or len(text) == 1)  # kaam, theek, door; but tha, ki
        initial_a = ch == "A" and i == 0 and followed  # aap, aaya, aadmi
        if only_vowel or initial_a or (double_long and followed):
            out.append(double)
        else:
            out.append(short)
    return "".join(out)


def romanise_word(word, final_nasal=True, double_long=False):
    """Romanise one Devanagari word.

    The two options are used by the noise step:
    final_nasal=False drops nasalisation at the end ("हैं" -> "hai", "में" -> "me").
    double_long=True doubles every long vowel that is not at the end ("शादी" -> "shaadi").
    """
    word = unicodedata.normalize("NFD", word)
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    syllables = _parse(word)
    if len(syllables) == 0:
        return ""
    _delete_schwas(syllables)

    text = _spell(syllables, final_nasal)
    text = _AH_RE.sub("eh", text)  # pehle, kehna, rehte

    nuclei = 0
    for syllable in syllables:
        if syllable.pronounced:
            nuclei += 1
    text = _spell_long_vowels(text, nuclei, double_long)
    return text.replace("H", "h")


def romanise(text, word_fn=romanise_word):
    """Romanise every Devanagari word in the text. Latin words, digits and symbols pass through."""
    text = _INVISIBLE_RE.sub("", text)
    text = text.translate(_PUNCTUATION)

    pieces = []
    position = 0
    for match in _WORD_RE.finditer(text):
        pieces.append(text[position : match.start()])
        pieces.append(word_fn(match.group()))
        position = match.end()
    pieces.append(text[position:])
    return "".join(pieces)

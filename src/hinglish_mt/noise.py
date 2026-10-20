"""Spelling noise for romanised Hindi, imitating how people type Hinglish.

The romaniser gives one spelling per word; real text has many ("nahi", "nhi",
"nahin"). Each operation copies a habit seen in PHINC, and the default
probabilities are set from PHINC train counts.

Operations, per word:
* nasal drop: leave out the nasal sound at the end ("hain" -> "hai", "mein" -> "me").
* long vowel doubling: "shadi" -> "shaadi" (never at the end of a word).
* shortening table: fixed variants of frequent words ("nahi" -> "nhi", "kyon" -> "kyu").
* vowel length: "aa"/"a", "ee"/"i", "oo"/"u", "ai" -> "e", final "i" -> "ee".
* vowel drop: remove one inner vowel ("karna" -> "krna").
* sound swap: letters that sound alike ("v"/"w", "ph" -> "f", "z" -> "j").
* repetition: stretch the last letter ("jaldi" -> "jaldiii").
"""

import re
from dataclasses import dataclass

from hinglish_mt.romanise import romanise, romanise_word


@dataclass(frozen=True)
class NoiseConfig:
    """Probability of each operation for one word."""

    nasal_drop: float = 0.5
    long_vowel_double: float = 0.4
    shortening_scale: float = 1.0  # multiplies the probabilities in SHORTENINGS
    vowel_length: float = 0.06
    vowel_drop: float = 0.05
    sound_swap: float = 0.04
    repeat: float = 0.01


# clean spelling: (probability of replacing it, variants). A variant listed twice is twice as likely.
SHORTENINGS = {
    "hai": (0.10, ("h", "h", "he")),
    "nahi": (0.17, ("nhi", "nhi", "nahin", "nai", "ni")),
    "mein": (0.10, ("mai", "m", "mei")),
    "main": (0.20, ("mai", "me")),
    "kya": (0.03, ("kia", "ky")),
    "aur": (0.17, ("or", "or", "r")),
    "bhi": (0.06, ("b", "b", "bi")),
    "kyon": (0.90, ("kyu", "kyu", "kyu", "kyun")),
    "kyo": (0.90, ("kyu",)),
    "kyonki": (0.80, ("kyunki", "kyuki")),
    "hoon": (0.50, ("hun", "hu")),
    "bahut": (0.45, ("bahot", "bahot", "bohot", "bht", "bhut")),
    "kuch": (0.10, ("kuchh",)),
    "ye": (0.15, ("yeh",)),
    "wo": (0.55, ("woh", "woh", "woh", "woh", "vo")),
    "to": (0.20, ("toh",)),
    "hum": (0.10, ("ham", "hm")),
    "kar": (0.07, ("kr",)),
    "karna": (0.10, ("krna",)),
    "par": (0.45, ("pe", "pe", "pe", "pe", "pe", "pe", "pe", "pe", "pe", "pr")),
    "raha": (0.08, ("rha",)),
    "rahe": (0.10, ("rhe",)),
    "rahi": (0.08, ("rhi",)),
    "accha": (0.50, ("achha", "achha", "acha", "achcha")),
    "theek": (0.18, ("thik",)),
    "pehle": (0.18, ("pahle", "pahle", "pahle", "pahle", "phle")),
    "saath": (0.30, ("sath",)),
    "baat": (0.10, ("bat",)),
    "aap": (0.10, ("ap",)),
    "liye": (0.04, ("lie",)),
    "gayi": (0.45, ("gayee", "gayee", "gayee", "gyi", "gai")),
    "sab": (0.04, ("sb",)),
    "phir": (0.35, ("fir",)),
    "pyaar": (0.60, ("pyar",)),
    "ke": (0.20, ("k",)),
    "ki": (0.10, ("k",)),
    "se": (0.03, ("say",)),
    "mujhe": (0.03, ("muje",)),
    "kaise": (0.08, ("kese",)),
    "sarkar": (0.50, ("sarkaar",)),
    "lekin": (0.10, ("lkin",)),
}

_VOWELS = "aeiou"

# Each rule is (pattern, replacement, probability). Habits that are rare in PHINC get a low probability.
_LENGTH_RULES = [
    (re.compile(r"aa"), "a", 1.0),
    (re.compile(r"ee"), "i", 1.0),
    (re.compile(r"oo"), "u", 1.0),
    (re.compile(r"ai(?=[^aeiou])"), "e", 0.5),
    (re.compile(r"(?<=[^aeiou])i$"), "ee", 0.25),
    (re.compile(r"(?<=[^aeiou])a$"), "aa", 0.25),
    (re.compile(r"(?<=[^aeiou])e$"), "ey", 0.1),
]
_SWAP_RULES = [
    (re.compile(r"ph"), "f", 1.0),
    (re.compile(r"z"), "j", 1.0),
    (re.compile(r"sh"), "s", 0.5),
    (re.compile(r"v"), "w", 0.5),
    (re.compile(r"w"), "v", 0.25),
    (re.compile(r"(?<=[bdgkt])h(?=[aeiou])"), "", 0.5),
    (re.compile(r"ksh"), "x", 0.25),
]


def _apply_one(word, rules, rng):
    """Pick one place where a rule fits, at random, and apply the rule with its probability."""
    options = []
    for pattern, replacement, probability in rules:
        for match in pattern.finditer(word):
            options.append((match, replacement, probability))
    if len(options) == 0:
        return word

    match, replacement, probability = rng.choice(options)
    if rng.random() >= probability:
        return word
    before = word[: match.start()]
    after = word[match.end() :]
    return before + replacement + after


def _drop_vowel(word, rng):
    """Remove one vowel that stands alone between two consonants. Mostly "a", never "o"."""
    spots = []
    for i in range(1, len(word) - 1):
        between_consonants = word[i - 1] not in _VOWELS and word[i + 1] not in _VOWELS
        if word[i] in "aeiu" and between_consonants:
            spots.append(i)
    if len(word) < 4 or len(spots) == 0:
        return word

    weights = []
    for i in spots:
        if word[i] == "a":
            weights.append(10)
        else:
            weights.append(1)
    i = rng.choices(spots, weights=weights)[0]
    return word[:i] + word[i + 1 :]


def noisy_word(word, rng, config=NoiseConfig()):
    """Return a noisy spelling of one clean romanised word (often the word itself)."""
    if word in SHORTENINGS:
        probability, variants = SHORTENINGS[word]
        if rng.random() < probability * config.shortening_scale:
            return rng.choice(variants)
    if len(word) < 3 or not word.isalpha():
        return word

    if rng.random() < config.vowel_length:
        word = _apply_one(word, _LENGTH_RULES, rng)
    if rng.random() < config.sound_swap:
        word = _apply_one(word, _SWAP_RULES, rng)
    if rng.random() < config.vowel_drop:
        word = _drop_vowel(word, rng)
    if rng.random() < config.repeat:
        extra_letters = rng.randint(1, 3)
        word = word + word[-1] * extra_letters
    return word


def noisy_romanise_word(word, rng, config=NoiseConfig()):
    """Romanise one Devanagari word with spelling noise."""
    keep_nasal = rng.random() >= config.nasal_drop
    double_long = rng.random() < config.long_vowel_double
    spelling = romanise_word(word, final_nasal=keep_nasal, double_long=double_long)
    return noisy_word(spelling, rng, config)


def noisy_romanise(text, rng, config=NoiseConfig()):
    """Romanise Devanagari text with spelling noise. Latin words in the text are left alone."""

    def noisy_spelling(word):
        return noisy_romanise_word(word, rng, config)

    return romanise(text, noisy_spelling)

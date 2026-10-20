"""Code-Mixing Index (Das and Gambäck, 2014), the measure the PHINC paper uses.

For one sentence with n tokens, u of them without a language, and w_i tokens
in language i:

    CMI = 100 * (1 - max(w_i) / (n - u))   if n > u, else 0

With two languages the larger one holds at least half of the tagged tokens,
so a sentence can score at most 50.
"""

from collections import Counter

from hinglish_mt.langid import EN, HI, OTHER


def language_counts(tags):
    counts = Counter()
    for tag in tags:
        if tag != OTHER:
            counts[tag] += 1
    return counts


def cmi(tags):
    counts = language_counts(tags)
    tagged = sum(counts.values())
    if tagged == 0:
        return 0.0
    largest = max(counts.values())
    return 100.0 * (1 - largest / tagged)


def switch_points(tags):
    """Number of language changes between neighbouring tokens that have a language."""
    languages = []
    for tag in tags:
        if tag != OTHER:
            languages.append(tag)
    switches = 0
    for i in range(1, len(languages)):
        if languages[i] != languages[i - 1]:
            switches += 1
    return switches


def corpus_cmi(tag_lists):
    """Summaries of sentence CMI over a corpus.

    mean_all averages over every sentence, mean_mixed over sentences that
    contain both languages, and pooled applies the formula to the token counts
    of the whole corpus.
    """
    scores = []
    mixed_scores = []
    all_tags = []
    switches = 0
    dominant = {HI: 0, EN: 0, "tie": 0, "none": 0}

    for tags in tag_lists:
        score = cmi(tags)
        scores.append(score)
        if score > 0:
            mixed_scores.append(score)
        all_tags.extend(tags)
        switches += switch_points(tags)

        counts = language_counts(tags)
        if len(counts) == 0:
            dominant["none"] += 1
        elif counts[HI] == counts[EN]:
            dominant["tie"] += 1
        elif counts[HI] > counts[EN]:
            dominant[HI] += 1
        else:
            dominant[EN] += 1

    n = len(tag_lists)
    if n == 0:
        n = 1
    n_tags = len(all_tags)
    if n_tags == 0:
        n_tags = 1

    if len(mixed_scores) > 0:
        mean_mixed = sum(mixed_scores) / len(mixed_scores)
    else:
        mean_mixed = 0.0

    tag_share = {}
    for tag in (HI, EN, OTHER):
        tag_share[tag] = all_tags.count(tag) / n_tags

    dominant_share = {}
    for key in (HI, EN, "tie", "none"):
        dominant_share[key] = dominant[key] / n

    return {
        "mean_all": sum(scores) / n,
        "mean_mixed": mean_mixed,
        "mixed_share": len(mixed_scores) / n,
        "pooled": cmi(all_tags),
        "mean_switch_points": switches / n,
        "tag_share": tag_share,
        "dominant_language_share": dominant_share,
    }

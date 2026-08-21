#!/usr/bin/env python3
"""Launch the SeamlessM4T-v2 stress experiment with the corrected WER normalizer.

This wrapper patches the project clean-WER module before importing the Seamless
pipeline.  It fixes the previous contraction regex that could incorrectly turn
ordinary words ending in ``nt`` (e.g. ``want`` -> ``wa not``) into artificial
WER errors.  All other pipeline logic remains unchanged.
"""

from __future__ import annotations

import math
import re
import unicodedata

import clean_wer_rescore as cwr


def normalize_asr_text_fixed(text: object) -> str:
    """Conservative ASR text normalization with safe contraction handling."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = cwr.WHISPER_SPECIAL.sub(" ", s)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")

    # Expand only genuine apostrophe-marked contractions here.  In particular,
    # the apostrophe is REQUIRED in n't; otherwise words such as want,
    # instrument, and establishment are corrupted.
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bshan't\b", "shall not", s)
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"'ll\b", " will", s)
    s = re.sub(r"'re\b", " are", s)
    s = re.sub(r"'ve\b", " have", s)
    s = re.sub(r"'m\b", " am", s)

    # Hyphenation/slashes become word boundaries; remaining punctuation becomes
    # whitespace rather than being deleted and accidentally joining tokens.
    s = re.sub(r"[-‐‑‒–—―/]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Decoder outputs may already contain split contraction fragments.
    s = re.sub(r"\b(i|you|he|she|it|we|they)\s+ll\b", r"\1 will", s)
    s = re.sub(r"\b(you|we|they)\s+re\b", r"\1 are", s)
    s = re.sub(r"\b(i|you|we|they)\s+ve\b", r"\1 have", s)
    s = re.sub(r"\bi\s+m\b", "i am", s)
    s = re.sub(r"\b(can)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(will)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(\w+)\s+n\s+t\b", r"\1 not", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_wer_fixed(reference: object, hypothesis: object):
    ref = normalize_asr_text_fixed(reference)
    hyp = normalize_asr_text_fixed(hypothesis)
    valid = ref not in cwr.INVALID_REFERENCES
    if not valid:
        return float("nan"), ref, hyp, False
    ref_words = ref.split()
    hyp_words = hyp.split()
    return (
        float(cwr._levenshtein_distance(ref_words, hyp_words) / len(ref_words)),
        ref,
        hyp,
        True,
    )


# Patch before importing the target pipeline. add_clean_wer() resolves clean_wer
# through clean_wer_rescore's module globals, so it will use this corrected
# function.  The Seamless pipeline will also import the corrected normalizer for
# output-concentration calculations.
cwr.normalize_asr_text = normalize_asr_text_fixed
cwr.clean_wer = clean_wer_fixed

# Safety assertions for the exact bug that was observed in the Whisper output.
assert normalize_asr_text_fixed("want") == "want"
assert normalize_asr_text_fixed("instrument") == "instrument"
assert normalize_asr_text_fixed("establishment") == "establishment"
assert normalize_asr_text_fixed("doesn't") == "does not"
assert clean_wer_fixed("She'll go", "she ll go")[0] == 0.0

import seamless_m4t_stress_pipeline as pipeline  # noqa: E402


if __name__ == "__main__":
    pipeline.main()

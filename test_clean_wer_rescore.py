import math

from clean_wer_rescore import clean_wer, normalize_asr_text


def test_contraction_normalization_matches_semantics():
    wer, ref, hyp, valid = clean_wer("It doesn't look good.", "it does not look good")
    assert valid
    assert wer == 0.0
    assert ref == hyp == "it does not look good"


def test_split_ll_contraction_is_normalized():
    wer, ref, hyp, valid = clean_wer("She'll be all right.", "she ll be all right")
    assert valid
    assert wer == 0.0
    assert ref == hyp == "she will be all right"


def test_hyphenated_compounds_are_token_boundaries():
    wer, _, _, valid = clean_wer(
        "He is of African-American descent.",
        "he is of african american descent",
    )
    assert valid
    assert wer == 0.0
    wer2, _, _, _ = clean_wer("excellent cigar-making skills", "excellent cigar making skills")
    assert wer2 == 0.0


def test_real_lexical_segmentation_difference_remains_an_error():
    wer, _, _, valid = clean_wer("Close all windows and logout", "close all windows and log out")
    assert valid
    assert wer > 0.0


def test_repetition_failure_remains_large_wer():
    wer, _, _, valid = clean_wer("four", "four four four")
    assert valid
    assert wer == 2.0


def test_placeholder_reference_is_invalid_not_huge_wer():
    wer, ref, _, valid = clean_wer("undefined", "everything on earth and he vanished")
    assert ref == "undefined"
    assert not valid
    assert math.isnan(wer)


def test_punctuation_becomes_boundary_not_deletion():
    assert normalize_asr_text("high-desert") == "high desert"
    assert normalize_asr_text("father's") == "father s"

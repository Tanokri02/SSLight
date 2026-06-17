from sslight.constants import Q8_TO_Q3, q8_to_q3, validate_sequence


def test_q8_to_q3_mapping():
    assert q8_to_q3("HGBITESC") == "HHEHCECC"


def test_q8_to_q3_unknown_defaults_to_coil():
    assert q8_to_q3("X") == "C"


def test_validate_sequence_rejects_ambiguous():
    import pytest

    with pytest.raises(ValueError, match="ambiguous"):
        validate_sequence("ACDX")


def test_validate_sequence_accepts_standard():
    validate_sequence("ACDEFGHIKLMNPQRSTVWY")

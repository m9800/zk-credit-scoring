import pytest
from scripts.optimize import choose, parse_estimate


def test_selects_by_estimated_cost_not_measured_proof_time():
    candidates = [
        {"valid": True, "columns": 40, "implementation": 0, "estimated_ns": 100, "proving_s": 1},
        {"valid": True, "columns": 40, "implementation": 1, "estimated_ns": 90, "proving_s": 100},
        {"valid": True, "columns": 10, "implementation": 0, "estimated_ns": 30, "proving_s": 1000},
        {"valid": False, "columns": 10, "implementation": 1, "estimated_ns": 1, "proving_s": 0},
    ]
    assert choose(candidates, 40)["implementation"] == 1
    assert choose(candidates)["columns"] == 10


def test_missing_or_nonpositive_estimates_are_rejected():
    with pytest.raises(ValueError):
        parse_estimate("Warning: no calibration")
    with pytest.raises(ValueError):
        parse_estimate("Total time cost (esitmated): 0 (ns)\nOptimal k: 11\n"
                       "Total number of rows: 100\nArithmetic-row k estimate: 7")


def test_estimate_parser_keeps_raw_row_estimate_and_validated_k():
    result = parse_estimate("Total time cost (esitmated): 123.5 (ns)\nOptimal k: 11\n"
                            "Total number of rows: 100\nArithmetic-row k estimate: 7")
    assert result == {"estimated_ns": 123.5, "k": 11, "arithmetic_rows": 100, "arithmetic_k": 7}

import pytest
from scripts.optimize import choose, parse_estimate


def test_selects_by_estimated_cost_not_measured_proof_time():
    candidates = [
        {
            "valid": True,
            "columns": 40,
            "implementation": 0,
            "estimated_ns": 100,
            "proving_s": 1,
        },
        {
            "valid": True,
            "columns": 40,
            "implementation": 1,
            "estimated_ns": 90,
            "proving_s": 100,
        },
        {
            "valid": True,
            "columns": 10,
            "implementation": 0,
            "estimated_ns": 30,
            "proving_s": 1000,
        },
        {
            "valid": False,
            "columns": 10,
            "implementation": 1,
            "estimated_ns": 1,
            "proving_s": 0,
        },
    ]
    assert choose(candidates, 40)["implementation"] == 1
    assert choose(candidates)["columns"] == 10


def test_missing_or_nonpositive_estimates_are_rejected():
    with pytest.raises(ValueError):
        parse_estimate("Warning: no calibration")
    with pytest.raises(ValueError):
        parse_estimate(
            "Total time cost (esitmated): 0 (ns)\nOptimal k: 11\n"
            "Total number of rows: 100\nArithmetic-row k estimate: 7"
        )


def test_estimate_parser_keeps_raw_row_estimate_and_validated_k():
    result = parse_estimate(
        "Total time cost (esitmated): 123.5 (ns)\nOptimal k: 11\n"
        "Total number of rows: 100\nArithmetic-row k estimate: 7"
    )
    assert result == {
        "estimated_ns": 123.5,
        "k": 11,
        "arithmetic_rows": 100,
        "arithmetic_k": 7,
    }


def test_multilayer_cost_order_checks_later_layers_and_skips_infeasible():
    from scripts.neural_optimizer import cheapest_verified

    candidates = [
        {"columns": 40, "implementations": [0, 1, 2], "k": 12, "estimated_ns": 10},
        {"columns": 10, "implementations": [0, 2, 1], "k": 12, "estimated_ns": 10},
        {"columns": 10, "implementations": [0, 0, 0], "k": 10, "estimated_ns": 1},
    ]
    checked = []

    def verify(c):
        checked.append(c)
        return c["k"] == 12

    assert cheapest_verified(candidates, verify) == candidates[0]
    assert checked == [candidates[2], candidates[0]]
    assert cheapest_verified(candidates, verify, 40) == candidates[0]


def test_neural_plans_expand_occurrences_not_layer_types():
    from scripts.neural_optimizer import plans

    model = {
        "layers": [{"layer_type": "FullyConnected"}] * 3
        + [{"layer_type": "Logistic", "implementation": 0}]
    }
    result = plans(model)
    assert len(result) == len(set(result)) == 216
    assert {p.implementations for p in result} >= {(0, 2, 5), (0, 5, 2), (5, 5, 5)}


def test_unknown_circuit_errors_are_not_capacity_failures():
    from scripts.neural_optimizer import capacity_failure

    assert capacity_failure("NotEnoughRowsAvailable")
    assert not capacity_failure("segmentation fault")
    assert not capacity_failure("assertion `left == right` failed")


def test_public_parameters_alone_rule_out_small_neural_domains():
    from scripts.neural_optimizer import public_domain_floor

    assert public_domain_floor({"tensors": [{"data": [0] * 2433}]}) == 12
    assert public_domain_floor({"tensors": [{"data": [0] * 5}]}) == 10

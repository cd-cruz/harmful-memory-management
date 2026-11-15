
from agentdriver.harmful_memory import harmful_memory_manager as hm

def test_classifier():
    assert hm.memory_classifier(-0.5) == "keep"
    assert hm.memory_classifier(0.25) == "keep"
    assert hm.memory_classifier(0.4) == "flag"
    assert hm.memory_classifier(0.8) == "reject"
    assert hm.memory_classifier(1.5) == "deletion_review"
    print("PASS: Memory classifier")


def test_delta_l2():
    delta = hm.calculate_delta_l2_error(3.5, 2.0)
    assert delta == 1.5
    print("PASS: Delta L2 calculation")


def test_validation():
    import numpy as np

    assert hm.validate_l2_error(np.float64(1.5))
    assert not hm.validate_l2_error(float("nan"))
    assert not hm.validate_l2_error(-1.0)
    print("PASS: L2 validation")


def test_deletion_review():
    hm.record_memory_result.clear()

    for delta in [1.2, 1.3, 1.5]:
        classification = hm.memory_classifier(delta)
        hm.record_memory(
            "test_memory",
            3.0,
            1.5,
            delta,
            classification
        )

    record = hm.record_memory_result["test_memory"]

    assert record["number_memory_retrieved"] == 3
    assert record["number_harmful_outcomes"] == 3

    should_delete = hm.memory_deletion_review(
        hm.record_memory_result,
        "test_memory"
    )

    assert should_delete is True
    print("PASS: Memory deletion review")

    hm.record_memory_result.clear()


if __name__ == "__main__":
    test_classifier()
    test_delta_l2()
    test_validation()
    test_deletion_review()

    print("All harmful memory manager tests passed!")

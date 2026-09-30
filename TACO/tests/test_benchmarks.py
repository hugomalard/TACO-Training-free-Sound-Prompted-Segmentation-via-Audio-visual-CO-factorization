from taco.check_results import compare_results
from taco.evaluate import SPECS


def test_benchmark_defaults_match_the_paper_runs():
    s4 = SPECS["s4"]
    assert s4.temporal and s4.batch_size == 1 and s4.iterations == 1800
    assert s4.drop_first_label is False and s4.seed is None
    ms3 = SPECS["ms3"]
    assert ms3.temporal is False and ms3.batch_size == 4 and ms3.seed == 42
    assert ms3.drop_first_label and ms3.shuffle and ms3.b3 == 125
    assert SPECS["ade_sp"].batch_size == 20 and SPECS["ade_sp"].iterations == 1800
    assert SPECS["avss"].iterations == 1800 and SPECS["avss"].image_tokens == "openclip"
    assert SPECS["avss"].batch_size == 18 and SPECS["avss"].semantic_classes == 71
    semantic = SPECS["ade_sp_semantic"]
    assert semantic.factorization == "conmf_txt" and semantic.iterations == 1800
    assert semantic.batch_size == 12 and semantic.dummy_fcclip


def test_checker_accepts_a_result_inside_two_std():
    results = {
        "benchmark": "ms3",
        "mean": {"miou": 43.15, "fscore": 47.5, "miou_u": 25.88, "fscore_u": 30.72},
    }
    assert compare_results(results) == []


def test_checker_rejects_a_result_outside_tolerance():
    results = {
        "benchmark": "s4",
        "mean": {"miou": 10.0, "fscore": 71.50, "miou_u": 29.68, "fscore_u": 41.91},
    }
    failures = compare_results(results)
    assert len(failures) == 1
    assert "s4.miou" in failures[0]

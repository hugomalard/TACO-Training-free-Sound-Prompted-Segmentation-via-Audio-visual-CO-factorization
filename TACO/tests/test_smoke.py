import math
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ms3_smoke_one_sample():
    metadata = Path("/tsi/dcase/avsbench/metadataExtended.csv")
    weights = Path("/home/ids/hmalard/fcclip_cocopan.pth")
    if not metadata.exists() or not weights.exists():
        pytest.skip("AVSBench metadata or FC-CLIP weights are not available")
    from taco.evaluate import run_benchmark

    summary = run_benchmark("ms3", runs=1, limit=1)
    for key in ("miou", "fscore", "miou_u", "fscore_u"):
        assert math.isfinite(summary["mean"][key])

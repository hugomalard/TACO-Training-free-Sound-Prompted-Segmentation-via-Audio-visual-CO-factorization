import torch

from taco.data import rewrite_frame_name, rgb2cls, v2_palette
from taco.metrics import SemanticMeter, batch_miou_fscore, max_threshold_scores, multi_iou


def test_frame_name_rewrite():
    assert rewrite_frame_name("a/b/foo_1.png") == "a/b/foo.mp4_1.png"
    assert rewrite_frame_name("video_12.jpg") == "video.mp4_12.jpg"


def test_ade_palette_decode():
    seg = torch.zeros(3, 2, 2)
    seg[0] = 20
    seg[1] = 4
    decoded = rgb2cls(seg)
    assert decoded.shape == (2, 2)
    assert torch.all(decoded == 2 * 256 + 4)


def test_v2_palette_first_colors():
    palette = v2_palette()
    assert palette.shape == (71, 3)
    assert palette[0].tolist() == [0, 0, 0]
    assert palette[1].tolist() == [128, 0, 0]


def test_threshold_sweep_picks_the_best_iou():
    gt = torch.zeros(4, 4)
    gt[0, 0] = 1
    pred = torch.full((4, 4), 0.2)
    pred[0, 0] = 0.7
    miou, fscore = max_threshold_scores([pred], [gt], thresholds=[0.5, 0.9])
    assert abs(miou - 100.0) < 1e-3
    assert fscore > 0.0


def test_multi_iou_perfect_overlap():
    pred = torch.zeros(2, 4, 4)
    pred[:, :2] = 1
    assert abs(float(multi_iou(pred, pred)) - 1.0) < 1e-5


def test_semantic_meter_perfect_class():
    prediction = torch.full((8, 8), 3.0)
    target = torch.full((8, 8), 3.0)
    ious, fscores, counts = batch_miou_fscore(prediction.view(1, 8, 8), target.view(1, 8, 8), nclass=5)
    assert int(counts.sum()) == 1
    assert float(ious[counts > 0]) == 1.0
    meter = SemanticMeter(5)
    meter.update(prediction, target)
    miou, fscore = meter.compute()
    assert abs(miou - 100.0) < 1e-3
    assert abs(fscore - 100.0) < 1e-3
    del fscores

"""Segmentation metrics."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

AVSBENCH_THRESHOLDS = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95,
]


def minmax(values):
    return (values - values.min()) / (values.max() - values.min())


def resize_map(values, size=224):
    if values.ndim == 2:
        values = values.view(1, 1, *values.shape)
    elif values.ndim == 3:
        values = values.unsqueeze(0)
    return F.interpolate(values.float(), size=(size, size), mode="bicubic", align_corners=True)


class Evaluator(object):
    """AVSBench thresholded mIoU and F-measure.

    F-measure is aggregated over the whole set before the precision-recall
    threshold is chosen.
    """

    def __init__(self) -> None:
        super(Evaluator, self).__init__()
        self.miou = []
        self.F = []
        self.N = 0
        self.metrics = ["mIoU", "Fmeasure"]

    def evaluate_batch(self, pred, target, thr=None) -> None:
        thrs = []
        for j in range(pred.size(0)):
            infer = pred[j]
            if thr is None:
                flat = infer.detach().cpu().numpy().flatten()
                thrs.append(np.sort(flat)[int(infer.shape[-2] * infer.shape[-1] / 2)])
            else:
                thrs.append(thr)
        infers, gts = pred.squeeze(1), target.squeeze(1)
        self.mask_iou(infers, gts, thrs)
        self.Eval_Fmeasure(infers, gts)

    def mask_iou(self, preds, targets, thrs, eps: float = 1e-7):
        assert len(preds.shape) == 3 and preds.shape == targets.shape
        self.N += 1
        count = preds.size(0)
        miou = 0.0
        for i in range(count):
            pred = preds[i].unsqueeze(0)
            target = targets[i].unsqueeze(0)
            num_pixels = pred.size(-1) * pred.size(-2)
            no_obj_flag = target.sum(2).sum(1) == 0
            pred = (pred > thrs[i]).int()
            inter = (pred * target).sum(2).sum(1)
            union = torch.max(pred, target).sum(2).sum(1)
            inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
            inter[no_obj_flag] = inter_no_obj[no_obj_flag]
            union[no_obj_flag] = num_pixels
            miou += (torch.sum(inter / (union + eps))).squeeze()
        miou = miou / count
        self.miou.append(miou.detach().cpu())
        return miou

    @staticmethod
    def _eval_pr(y_pred, y, num):
        prec = torch.zeros(num, device=y_pred.device)
        recall = torch.zeros(num, device=y_pred.device)
        thlist = torch.linspace(0, 1 - 1e-10, num, device=y_pred.device)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i] = tp / (y_temp.sum() + 1e-20)
            recall[i] = tp / (y.sum() + 1e-20)
        return prec, recall

    def Eval_Fmeasure(self, pred, gt, pr_num: int = 255):
        count = pred.size(0)
        beta2 = 0.3
        avg_f, img_num = 0.0, 0
        score = torch.zeros(pr_num, device=pred.device)
        for img_id in range(count):
            if torch.sum(gt[img_id]) == 0.0:
                continue
            prec, recall = self._eval_pr(pred[img_id], gt[img_id], pr_num)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0
            avg_f += f_score
            img_num += 1
            score = avg_f / img_num
            self.F.append(f_score.detach().cpu().numpy())
        if img_num == 0:
            return 0.0
        return score.max().item()

    def finalize_mIoU(self) -> float:
        return np.sum(np.array(self.miou)) / self.N

    def finalize_Fmeasure(self) -> float:
        return np.max(np.mean(self.F, axis=0))

    def finalize(self) -> Tuple[List[str], Dict[str, float]]:
        return self.metrics, {
            self.metrics[0]: self.finalize_mIoU() * 100,
            self.metrics[1]: self.finalize_Fmeasure() * 100,
        }


def max_threshold_scores(predictions, targets, thresholds=None):
    """Max mIoU and F, in percent, over the AVSBench threshold grid.

    ``predictions`` and ``targets`` are lists of (H, W) maps. F-measure ignores
    the outer threshold and sweeps its own precision-recall curve.
    """
    thresholds = AVSBENCH_THRESHOLDS if thresholds is None else thresholds
    evaluators = [Evaluator() for _ in thresholds]
    for pred, gt in zip(predictions, targets):
        pred_b = pred.detach().float().cpu().reshape(1, *pred.shape[-2:])
        gt_b = gt.detach().float().cpu().reshape(1, 1, *gt.shape[-2:])
        for evaluator, thr in zip(evaluators, thresholds):
            evaluator.evaluate_batch(pred_b, gt_b, thr)
    miou, fscore = [], []
    for evaluator in evaluators:
        _, scores = evaluator.finalize()
        miou.append(scores["mIoU"])
        fscore.append(scores["Fmeasure"])
    return float(np.max(miou)), float(np.max(fscore))


def multi_iou(prediction, target, k=20):
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu() > 0.5
    thresholds = torch.linspace(prediction.min(), prediction.max(), k)
    hard_pred = prediction.unsqueeze(0) > thresholds.reshape(k, 1, 1, 1, 1)
    target = torch.broadcast_to(target.unsqueeze(0), hard_pred.shape)
    intersection = torch.logical_and(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.logical_or(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.where(union == 0, torch.tensor(1.0), union)
    iou_scores = intersection / union
    best_iou, _ = torch.max(iou_scores, dim=0)
    return best_iou


def per_class_detection_scores(predictions, targets, class_ids):
    """Mean over classes of best-threshold IoU and average precision, in percent."""
    from torchmetrics.functional.classification import binary_average_precision

    predictions = torch.stack([p.detach().float().cpu().reshape(*p.shape[-2:]) for p in predictions])
    targets = torch.stack([t.detach().float().cpu().reshape(*t.shape[-2:]) for t in targets])
    class_ids = torch.as_tensor(class_ids)
    unique = torch.unique(class_ids)
    ious = []
    aps = []
    for class_id in unique:
        selected = class_ids == class_id
        ious.append(multi_iou(predictions[selected], targets[selected]))
        aps.append(
            binary_average_precision(
                predictions[selected].flatten(),
                targets[selected].flatten().int(),
            )
        )
    return float(torch.stack(ious).mean()) * 100, float(torch.stack(aps).mean()) * 100


def batch_miou_fscore(output, target, nclass, beta2=0.3):
    """Histogram mIoU and F-score for semantic segmentation."""
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = (output + 1).float()
    target = target.float() + 1
    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    ious = torch.zeros(nclass)
    fscores = torch.zeros(nclass)
    cls_count = torch.zeros(nclass)
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        iou = area_inter.float() / (2.220446049250313e-16 + area_union.float())
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1
        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.0
        fscores += fscore
    return ious, fscores, cls_count


class SemanticMeter(object):
    def __init__(self, nclass):
        self.nclass = nclass
        self.ious = torch.zeros(nclass)
        self.fscores = torch.zeros(nclass)
        self.counts = torch.zeros(nclass)

    def update(self, prediction, target):
        pred = prediction.detach().float().cpu().reshape(1, *prediction.shape[-2:])
        gt = target.detach().float().cpu().reshape(1, *target.shape[-2:])
        ious, fscores, counts = batch_miou_fscore(pred, gt, self.nclass)
        self.ious += ious
        self.fscores += fscores
        self.counts += counts

    def compute(self):
        present = self.counts > 0
        miou = (self.ious[present] / self.counts[present]).mean().item() * 100
        fscore = (self.fscores[present] / self.counts[present]).mean().item() * 100
        return float(miou), float(fscore)

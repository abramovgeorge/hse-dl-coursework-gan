import torch
from torchmetrics.classification import F1Score

from src.metrics.base_metric import BaseMetric


class F1Metric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Metric wrapper for f1-score

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = F1Score(task="binary")
        self.metric = metric.to(device)
        self.name = "F1"

    def __call__(self, m_logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            m_logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(m_logits, labels.max(axis=1).indices.reshape(-1, 1))

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class GANInferencer(BaseTrainer):
    """
    GAN Inferencer class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference (i.e. sample and save specified amount of specified type of objects).
        Returns:
            stub (dict): empty dict for inference.py consistency
        """
        size = self.cfg_trainer.get("number_of_samples", 1)
        s_type = self.cfg_trainer.get("type", "mixed")
        type_vec = torch.randint(self.model.n_class, (size,)).to(self.device)
        if s_type != "mixed":
            type_vec = torch.full((size,), s_type).to(self.device)
        noise = torch.randn(size, self.model.noise_dim).to(self.device)
        cond = F.one_hot(type_vec, num_classes=self.model.n_class)
        batch = {"noise": noise, "cond": cond}
        fake_data = self.model.generator(**batch)["fake_data"]

        fake_data = self.batch_transforms["inference"]["data_object"].inverse(fake_data)

        output = pd.DataFrame(fake_data.cpu().detach().numpy())
        output.to_csv(self.save_path / f"{str(s_type)}.csv", index=False)

        return dict()

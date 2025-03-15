import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.trainer.gan_inferencer import GANInferencer


class CTGANInferencer(GANInferencer):
    """
    CTGAN Inferencer class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def run_inference(self):
        """
        Run inference (i.e. sample and save specified amount of specified type of objects).
        Returns:
            stub (dict): empty dict for inference.py consistency
        """
        size = self.cfg_trainer.get("number_of_samples", 1)
        s_type = self.cfg_trainer.get("type", "mixed")
        type_vec = torch.randint(2, (size,)).to(self.device)
        if s_type != "mixed":
            type_vec = torch.full((size,), s_type).to(self.device)
        noise = torch.randn(size, self.model.noise_dim).to(self.device)
        conds = []
        for type_value in type_vec:
            cond = self.evaluation_dataloaders["train"].get_cond_vector(
                self.cfg_trainer.get("target_class"), type_value
            )
            conds.append(cond)
        conds = torch.stack(conds).to(self.device)
        batch = {"noise": noise, "cond": conds}
        fake_data = self.model.generator(**batch)["fake_data"]

        fake_data = self.batch_transforms["inference"]["data_object"].inverse(fake_data)

        output = pd.DataFrame(fake_data.cpu().detach().numpy())
        output.to_csv(self.save_path / f"{str(s_type)}.csv", index=False)

        return dict()

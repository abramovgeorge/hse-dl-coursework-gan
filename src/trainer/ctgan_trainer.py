import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.gan_trainer import GANTrainer


class CTGANTrainer(GANTrainer):
    """
    CTGANTrainer class. Defines the logic of batch logging and processing for CTGAN.
    """

    def _train_discriminator(self, data_object, **batch):
        """
        One iteration of training discriminator

        Args:
            data_object (Tensor): data_object tensor in batch
        Returns:
            discriminator_loss (dict): dict, containing discriminator loss
        """
        # we train discriminator and generator on the same cond vector
        self.optimizers["discriminator"].zero_grad()

        batch["data"] = data_object
        real_logits = self.model.discriminator(**batch)
        batch["real_logits"] = real_logits["logits"]

        fake_data = self.model.generator(**batch)["fake_data"].detach()
        batch["data"] = fake_data
        fake_logits = self.model.discriminator(**batch)
        batch["fake_logits"] = fake_logits["logits"]

        loss = self.criterion.discriminator(**batch)
        batch.update(loss)
        batch["discriminator_loss"].backward()
        self._clip_grad_norm()
        self.optimizers["discriminator"].step()
        if self.lr_schedulers["discriminator"] is not None:
            self.lr_schedulers["discriminator"].step()
        return {"discriminator_loss": batch["discriminator_loss"]}

    def _train_generator(self, fake_labels, **batch):
        """
        One iteration of training generator

        Args:
            fake_labels (Tensor): random fake labels tensor in batch
        Returns:
            generator_loss (dict): dict, containing generator loss
        """
        # we train discriminator and generator on the same cond vector
        self.optimizers["generator"].zero_grad()

        fake_data = self.model.generator(**batch)
        batch.update(fake_data)
        batch["data"] = batch["fake_data"]
        fake_logits = self.model.discriminator(**batch)
        batch["fake_logits"] = fake_logits["logits"]

        loss = self.criterion.generator(**batch)
        batch.update(loss)
        loss = self.criterion.conditional(**batch)
        batch.update(loss)
        batch["generator_loss"] += batch["conditional_loss"]
        batch["generator_loss"].backward()
        self._clip_grad_norm()
        self.optimizers["generator"].step()
        if self.lr_schedulers["generator"] is not None:
            self.lr_schedulers["generator"].step()
        return {
            "generator_loss": batch["generator_loss"],
            "conditional_loss": batch["conditional_loss"],
        }

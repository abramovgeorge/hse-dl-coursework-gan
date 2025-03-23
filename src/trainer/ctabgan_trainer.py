import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.tgan_trainer import TGANTrainer


class CTABGANTrainer(TGANTrainer):
    """
    CTABGANTrainer class. Defines the logic of batch logging and processing for CTABGAN.
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

    def _train_generator(self, data_object, fake_labels, **batch):
        """
        One iteration of training generator

        Args:
            data_object (Tensor): data_object tensor in batch
            fake_labels (Tensor): random fake labels tensor in batch
        Returns:
            generator_loss (dict): dict, containing generator loss
        """
        self.optimizers["generator"].zero_grad()

        batch["data"] = data_object
        real_features = self.model.discriminator(**batch)["features"]

        fake_data = self.model.generator(**batch)
        batch.update(fake_data)
        batch["data"] = batch["fake_data"]
        d_output = self.model.discriminator(**batch)
        batch["fake_logits"] = d_output["logits"]

        batch["data"] = batch["fake_data"][:, : -self.model.n_class]  # drop labels
        fake_c_logits = self.model.classifier(**batch)
        batch["generated_labels"] = torch.argmax(
            batch["fake_data"][:, -self.model.n_class :], dim=1
        )
        batch["fake_c_logits"] = fake_c_logits["c_logits"]

        fake_features = d_output["features"]
        batch["real_features"] = real_features
        batch["fake_features"] = fake_features

        loss = self.criterion.generator(**batch)
        batch.update(loss)
        loss = self.criterion.classification(**batch)
        batch.update(loss)
        loss = self.criterion.information(**batch)
        batch.update(loss)
        loss = self.criterion.conditional(**batch)
        batch.update(loss)
        batch["generator_loss"] += (
            batch["classification_loss"]
            + batch["information_loss"]
            + batch["conditional_loss"]
        )
        batch["generator_loss"].backward()
        self._clip_grad_norm()
        self.optimizers["generator"].step()
        if self.lr_schedulers["generator"] is not None:
            self.lr_schedulers["generator"].step()
        return {
            "generator_loss": batch["generator_loss"],
            "classification_loss": batch["classification_loss"],
            "information_loss": batch["information_loss"],
            "conditional_loss": batch["conditional_loss"],
        }

    def _train_classifier(self, data_object, **batch):
        """
        One iteration of training classifier

        Args:
            data_object (Tensor): random noise tensor in batch
        Returns:
            output (dict): dict, containing classifier loss and classifier logits for the metrics
        """
        self.optimizers["classifier"].zero_grad()
        data_without_labels = data_object[:, : -self.model.n_class]
        logits = self.model.classifier(data_without_labels)
        batch.update(logits)
        loss = self.criterion.classifier(**batch)
        batch.update(loss)
        batch["classifier_loss"].backward()
        self._clip_grad_norm()
        self.optimizers["classifier"].step()
        if self.lr_schedulers["classifier"] is not None:
            self.lr_schedulers["classifier"].step()
        return {
            "classifier_loss": batch["classifier_loss"],
            "m_logits": batch["c_logits"],
        }

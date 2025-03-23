import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.gan_trainer import GANTrainer


class TGANTrainer(GANTrainer):
    """
    TGANTrainer class. Defines the logic of batch logging and processing for Tabular GAN.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train:
            for _ in range(self.n_d_steps):
                random_outputs = self._generate_noise_and_labels(**batch)
                batch.update(random_outputs)
                loss = self._train_discriminator(**batch)
                batch.update(loss)
                loss = self._train_classifier(**batch)
                batch.update(loss)
            random_outputs = self._generate_noise_and_labels(**batch)
            batch.update(random_outputs)
            loss = self._train_generator(**batch)
            batch.update(loss)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        # for logger debug in base trainer
        batch["loss"] = torch.abs(batch["discriminator_loss"] - batch["generator_loss"])
        return batch

    def _train_generator(self, data_object, labels, fake_labels, **batch):
        """
        One iteration of training generator

        Args:
            data_object (Tensor): data_object tensor in batch
            labels (Tensor): labels of data in batch
            fake_labels (Tensor): random fake labels tensor in batch
        Returns:
            generator_loss (dict): dict, containing generator loss
        """
        self.optimizers["generator"].zero_grad()

        batch["cond"] = labels
        batch["data"] = data_object
        real_features = self.model.discriminator(**batch)["features"]

        batch["cond"] = fake_labels
        fake_data = self.model.generator(**batch)["fake_data"]
        batch["data"] = fake_data
        d_output = self.model.discriminator(**batch)
        batch["fake_logits"] = d_output["logits"]

        batch["data"] = fake_data[:, :-1]  # drop labels
        fake_c_logits = self.model.classifier(**batch)
        batch["generated_labels"] = fake_data[:, -1]
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
        batch["generator_loss"] += (
            batch["classification_loss"] + batch["information_loss"]
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
        data_without_labels = data_object[:, :-1]
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

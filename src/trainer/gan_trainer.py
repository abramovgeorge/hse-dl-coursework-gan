import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    """
    GANTrainer class. Defines the logic of batch logging and processing for GANs.
    """

    def __init__(self, config, *args, **kwargs):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizers (dict[Optimizer]): optimizers for the model.
            lr_schedulers (dict[LRScheduler]): learning rate schedulers for the
                optimizers.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        super().__init__(config=config, *args, **kwargs)
        self.n_d_steps = config.trainer.get("n_d_steps", 1)

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

    def _train_discriminator(self, data_object, labels, fake_labels, **batch):
        """
        One iteration of training discriminator

        Args:
            data_object (Tensor): data_object tensor in batch
            labels (Tensor): labels of data in batch
            fake_labels (Tensor): random fake labels tensor in batch
        Returns:
            discriminator_loss (dict): dict, containing discriminator loss
        """
        self.optimizers["discriminator"].zero_grad()

        batch["data"] = data_object
        batch["cond"] = labels
        real_logits = self.model.discriminator(**batch)
        batch["real_logits"] = real_logits["logits"]

        fake_data = self.model.generator(**batch)["fake_data"].detach()
        batch["data"] = fake_data
        batch["cond"] = fake_labels
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
        self.optimizers["generator"].zero_grad()

        batch["cond"] = fake_labels
        fake_data = self.model.generator(**batch)["fake_data"]
        batch["data"] = fake_data
        fake_logits = self.model.discriminator(**batch)
        batch["fake_logits"] = fake_logits["logits"]

        loss = self.criterion.generator(**batch)
        batch.update(loss)
        batch["generator_loss"].backward()
        self._clip_grad_norm()
        self.optimizers["generator"].step()
        if self.lr_schedulers["generator"] is not None:
            self.lr_schedulers["generator"].step()
        return {"generator_loss": batch["generator_loss"]}

    def _generate_noise_and_labels(self, data_object, **batch):
        """
        Generate random noise from normal distribution and add it to batch

        Args:
            data_object (Tensor): data_object tensor in batch (needed only for batch size)
        Returns:
            noise (Tensor): random noise tensor in batch
            fake labels (Tensor): random fake labels tensor in batch
        """
        return {
            "noise": torch.randn(data_object.shape[0], self.model.noise_dim).to(
                self.device
            ),
            "fake_labels": F.one_hot(
                torch.randint(self.model.n_class, (data_object.shape[0],)).to(
                    self.device
                ),
                num_classes=self.model.n_class,
            ),
        }

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass

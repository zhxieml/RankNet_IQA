import collections
import os

import numpy as np
import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, criterion, metric_fns, optimizer, config, device, train_dataloader,
                 valid_dataloader=None, lr_scheduler=None, logger=None):
        self._model = model
        self._criterion = criterion
        self._metric_fns = metric_fns
        self._optimizer = optimizer
        self._config = config
        self._device = device
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._lr_scheduler = lr_scheduler
        self._logger = logger

        # Parse the config.
        trainer_config = self._config["trainer"]
        self._resume = self._config.resume
        self._save_dirname = self._config.save_dirname
        self._num_epoch = trainer_config["num_epoch"]
        self._save_period = trainer_config["save_period"]
        self._early_stop = trainer_config["early_stop"]
        self._start_epoch_idx = 0
        self._best_valid_metrics = None

        if self._resume is not None:
            self._resume_checkpoint(self._resume)

    def _save_checkpoint(self, epoch_idx):
        checkpoint_filename = os.path.join(self._save_dirname, "checkpoint_epoch{}.pth".format(epoch_idx))
        state = {
            "epoch": epoch_idx,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "best_valid_metrics": self._best_valid_metrics,
            "config": self._config
        }

        torch.save(state, checkpoint_filename)

    def _resume_checkpoint(self, checkpoint_filename):
        state = torch.load(checkpoint_filename)

        if state["config"]["arch"]["type"] != self.config["arch"]["type"]:
            raise Exception("Failed at resuming checkpoint: conflicting model types.")
        if state["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            raise Exception("Failed at resuming checkpoint: conflicting optimizer types.")

        self._model.load_state_dict(state["state_dict"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._best_valid_metrics = state["best_valid_metrics"]

        return state["epoch"] + 1

    def _train_epoch(self, epoch_idx):
        # Set the train mode.
        self._model.train()
        progress_bar = tqdm(enumerate(self._train_dataloader))

        for _, ((first_imgs_batch, second_imgs_batch), label_batch) in progress_bar:
            # Forward.
            first_imgs_batch, second_imgs_batch, label_batch = first_imgs_batch.to(self._device), second_imgs_batch.to(self._device), label_batch.to(self._device)
            outputs = self._model(first_imgs_batch, second_imgs_batch)
            loss = self._criterion(outputs, label_batch)

            # Backword.
            self._model.zero_grad()
            loss.backward()
            self._optimizer.step()

            # Record.
            progress_bar.set_description("[Epoch {}/{}] Loss: {}".format(epoch_idx + 1, self._num_epoch, loss.item()))

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _valid_epoch(self, epoch_idx):
        # Set the eval mode.
        self._model.eval()
        valid_metrics = collections.defaultdict(list)

        with torch.no_grad():
            for _, (data_batch, label_batch) in enumerate(self._valid_dataloader):
                data_batch, label_batch = data_batch.to(self._device), label_batch.to(self._device)
                outputs = self._model.predict(data_batch)

                for metric_name, metric_fn in self._metric_fns.items():
                    metric = metric_fn(outputs, label_batch)
                    valid_metrics[metric_name].append(metric.item())

        # Record.
        if self._logger is not None:
            self._logger.debug("############ Validation [{}/{}] ############".format(epoch_idx + 1, self._num_epoch))
            for metric_name, metric_list in valid_metrics.items():
                self._logger.debug("{}: {:.4f}".format(metric_name, np.mean(metric_list)))
            self._logger.debug("\n")

        return valid_metrics

    def train(self):
        not_improved_count = 0

        for epoch_idx in range(self._start_epoch_idx, self._num_epoch):
            self._train_epoch(epoch_idx)
            valid_metrics = self._valid_epoch(epoch_idx)

            # Check whether to stop early (assume the higher the better).
            first_try = self._best_valid_metrics is None
            improved = first_try or np.all([valid_metrics[metric_name] > self._best_valid_metrics[metric_name] for metric_name in valid_metrics])
            if improved:
                not_improved_count = 0
                self._best_valid_metrics = valid_metrics
            else:
                not_improved_count += 1

            if not_improved_count > self._early_stop:
                print("Early stopped.")
                break

            if epoch_idx % self._save_period == 0:
                self._save_checkpoint(epoch_idx)
import collections
import os

import numpy as np
import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, criterion, metric_fns, optimizer, config, device,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None):
        self._model = model
        self._criterion = criterion
        self._metric_fns = metric_fns
        self._optimizer = optimizer
        self._config = config
        self._device = device
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._lr_scheduler = lr_scheduler

        # Parse the config.
        trainer_config = self._config["trainer"]
        self._enable_log = self._config.enable_log
        self._resume = self._config.resume
        self._num_epoch = trainer_config["num_epoch"]
        self._early_stop = trainer_config["early_stop"]
        self._start_epoch_idx = 0
        self._best_valid_metrics = None

        # Setup logging.
        if self._enable_log:
            self._save_dirname = self._config.save_dirname
            self._save_period = trainer_config["save_period"]
            self._logger = config.get_logger("trainer", trainer_config["verbosity"])

        # Resume checkpoint.
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

        if state["config"]["arch"]["type"] != self._config["arch"]["type"]:
            raise Exception("Failed at resuming checkpoint: conflicting model types.")
        if state["config"]["optimizer"]["type"] != self._config["optimizer"]["type"]:
            raise Exception("Failed at resuming checkpoint: conflicting optimizer types.")

        self._model.load_state_dict(state["state_dict"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._best_valid_metrics = state["best_valid_metrics"]
        self._start_epoch_idx = state["epoch"] + 1

        return state["epoch"] + 1

    def _train_epoch(self, epoch_idx):
        # Set the train mode.
        self._model.train()

        for batch_idx, ((first_data_batch, second_data_batch), label_batch) in enumerate(self._train_dataloader):
            # Forward.
            first_data_batch, second_data_batch, label_batch = first_data_batch.to(self._device), second_data_batch.to(self._device), label_batch.to(self._device)
            outputs = self._model(first_data_batch, second_data_batch)
            loss = self._criterion(outputs, label_batch)

            # Backword.
            self._model.zero_grad()
            loss.backward()
            self._optimizer.step()

            # Record.
            if self._enable_log and not batch_idx % 10:
                self._logger.info("[Epoch {}/{} Batch {}] Loss: {:.4f}".format(epoch_idx + 1, self._num_epoch, batch_idx, loss.item()))

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _valid_epoch(self, epoch_idx):
        # Set the eval mode.
        self._model.eval()
        valid_metrics = collections.defaultdict(list)

        with torch.no_grad():
            for _, ((first_data_batch, second_data_batch), label_batch) in enumerate(self._valid_dataloader):
                # Forward.
                first_data_batch, second_data_batch, label_batch = first_data_batch.to(self._device), second_data_batch.to(self._device), label_batch.to(self._device)
                outputs = self._model(first_data_batch, second_data_batch)
                outputs, label_batch = outputs.cpu().numpy()[:, 0], label_batch.cpu().numpy()[:, 0]

                for metric_name, metric_fn in self._metric_fns.items():
                    metric = metric_fn(outputs, label_batch)
                    valid_metrics[metric_name].append(metric.item())

        for metric_name, metric_list in valid_metrics.items():
            valid_metrics[metric_name] = np.mean(metric_list)

        # Record.
        if self._enable_log and self._logger is not None:
            self._logger.info("############ Validation [{}/{}] ############".format(epoch_idx + 1, self._num_epoch))
            for metric_name, mean_metric in valid_metrics.items():
                self._logger.info("{}: {:.4f}".format(metric_name, mean_metric))

        return valid_metrics

    def train(self):
        not_improved_count = 0

        for epoch_idx in range(self._start_epoch_idx, self._num_epoch):
            # Train.
            self._train_epoch(epoch_idx)

            # Validation.
            if self._valid_dataloader:
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
                    if self._enable_log: self._logger.info("Early stopped.")
                    break

            if self._enable_log and epoch_idx % self._save_period == 0:
                self._save_checkpoint(epoch_idx)
import logging
import os
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from nlplay.models.pytorch.lr_finder import LRFinder
from nlplay.models.pytorch.utils import set_seed
from nlplay.models.pytorch.metrics import compute_accuracy
from nlplay.utils.utils import get_elapsed_time


class PytorchModelTrainer(object):
    def __init__(
        self,
        model=None,
        criterion=None,
        optimizer=None,
        train_ds: Dataset = None,
        test_ds: Dataset = None,
        val_ds: Dataset = None,
        batch_size: int = 64,
        n_workers: int = 1,
        epochs: int = 5,
        model_output_folder: Path = "",
        early_stopping=True,
        early_stopping_patience: int = 3
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_workers = n_workers
        self.n_epochs = epochs

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.train_dl = None
        self.test_dl = None
        self.val_dl = None

        self.apply_early_stopping = early_stopping
        self.early_stop = False
        self.best_score = None
        self.best_epoch = -1
        self.es_counter = 0
        self.es_patience = early_stopping_patience
        self.es_improvement_delta = 0
        self.model_output_folder = model_output_folder
        logging.getLogger(__name__)

    def train(self, seed=42, check_dl=True, run_lr_finder=False):

        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        if self.test_ds is not None:
            self.test_dl = DataLoader(
                self.test_ds, batch_size=self.batch_size, shuffle=False
            )
        if self.val_ds is not None:
            self.val_dl = DataLoader(
                self.val_ds, batch_size=self.batch_size, shuffle=False
            )

        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

        if run_lr_finder:
            logging.info("LR Finder Running....")
            lr_finder = LRFinder(
                self.model, self.optimizer, criterion=self.criterion, device=device
            )
            lr_finder.range_test(self.train_dl, end_lr=100, num_iter=100)
            lr_finder.plot(show=False, folder_path="../../")
            logging.info("LR Finder Run Completed....")

        # Checking the dataloaders
        if check_dl:
            for data, labels in self.train_dl:
                logging.info("----------------------------------")
                logging.info("---       DATALOADER INFO      ---")
                logging.info("----------------------------------")
                logging.info("Train DataLoader Details:")
                logging.info("   batch dimensions: {}".format(data.shape))
                logging.info("   label dimensions: {}".format(labels.shape))
                break

            for data, labels in self.test_dl:
                logging.info("Test DataLoader Details:")
                logging.info("   batch dimensions: {}".format(data.shape))
                logging.info("   label dimensions: {}".format(labels.shape))
                break

        logging.info("----------------------------------")
        logging.info("---       MODEL TRAINING       ---")
        logging.info("----------------------------------")
        model_parameters_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        n_iters = len(self.train_ds) / self.batch_size
        logging.info("Number of iterations/epoch : {}".format(n_iters))
        log_interval = 10

        start_time = time.time()
        # Loop over epochs
        for epoch in range(self.n_epochs):
            self.model.train()
            for batch_index, (batch_train_data, batch_train_labels) in enumerate(
                self.train_dl
            ):

                # transfer data to target device
                batch_train_data = batch_train_data.to(device)
                batch_train_labels = batch_train_labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(batch_train_data,)
                loss = self.criterion(outputs, batch_train_labels)
                loss.backward()
                self.optimizer.step()

                # Report intermediate loss value after a certain amount of batches
                if (batch_index + 1) % log_interval == 0:
                    logging.info(
                        "   Info | Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.6f"
                        % (epoch + 1, self.n_epochs, batch_index + 1, n_iters, loss)
                    )

            # End of epoch - Evaluate the model performance
            self.model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference
                logging.info(
                    "Epoch: %03d/%03d | Train Accuracy: %.6f"
                    % (
                        epoch + 1,
                        self.n_epochs,
                        compute_accuracy(self.model, self.train_dl, device=device),
                    )
                )
                test_acc = compute_accuracy(self.model, self.test_dl, device=device)
                logging.info(
                    "Epoch: %03d/%03d | Test accuracy: %.6f"
                    % (epoch + 1, self.n_epochs, test_acc)
                )
                logging.info("Time elapsed: {}".format(get_elapsed_time(start_time)))

                # early stopping & checkpoint
                current_score = test_acc
                if self.best_score is None:
                    self.best_score = current_score.to(torch.device("cpu")).numpy()
                    self.best_epoch = epoch + 1
                    self.save_checkpoint()
                elif (
                    self.apply_early_stopping
                    and current_score < self.best_score + self.es_improvement_delta
                ):
                    self.es_counter += 1
                    logging.info(
                        f"EarlyStopping patience counter: {self.es_counter} out of {self.es_patience}"
                    )
                    if self.es_counter >= self.es_patience:
                        self.early_stop = True
                        logging.warning("/!\ Early stopping model training /!\ ")
                        break
                else:
                    self.best_score = current_score
                    self.best_epoch = epoch + 1
                    self.save_checkpoint()
                    self.es_counter = 0

        # Final results
        logging.info("----------------------------------")
        logging.info("---          SUMMARY           ---")
        logging.info("----------------------------------")
        logging.info("Number of model parameters : {}".format(model_parameters_count))
        logging.info("Total Training Time: {}".format(get_elapsed_time(start_time)))
        logging.info("Total Time: {}".format(get_elapsed_time(start_time)))
        logging.info(
            "Best Epoch: {} - Accuracy Score: {:.6f}".format(
                self.best_epoch, self.best_score
            )
        )
        logging.info("----------------------------------")

    def save_checkpoint(self):
        """Saves model when validation loss decrease."""
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.model_output_folder,
                "{}_checkpoint.pt".format(self.model.__class__.__name__),
            ),
        )

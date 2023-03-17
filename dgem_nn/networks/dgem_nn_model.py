import logging
from typing import Iterable

import torch
import tqdm as tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from torch.nn import DataParallel, Module
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from dgem_nn.data_processing.omics_dataset import OmicsDataset, get_weighted_sampler

logger = logging.getLogger(__name__)


def evaluate(y_true: Iterable, y_pred: Iterable, threshold: float = 0.5):
    """
    calculate metrics during training
    """
    y_pred_s = [1 if i >= threshold else 0 for i in y_pred]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    return {
        "f1": round(f1_score(y_true, y_pred_s), 3),
        "precision": round(precision_score(y_true, y_pred_s), 3),
        "recall": round(recall_score(y_true, y_pred_s), 3),
        "auc": auc(fpr, tpr),
    }


class DgemNN:
    def __init__(self, model_type: Module, model_path: str):
        """
        This is the DGEM NN class. It implements training, evaluation and prediction
        :param model_type: The NN architecture to be used
        :param model_path: where to save/load model
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_path = model_path

    def train(
            self,
            train_dataset: OmicsDataset,
            val_dataset: OmicsDataset,
            batch_size: int,
            lr: float,
            num_epochs: int,
            warmup_steps: int,
            eval_every_steps: int = -1,
            gradient_accumulation_steps: int = 1,
            balance_batches: bool = True,
            initial_patience: int = 5,
            internal_layer_size: int = 1024,
    ):
        """
        Train a model
        :param balance_batches: use a DataLoader that loads balanced positive/negative
        samples
        :param initial_patience: if not improving on dev set metrics, this does early
        stopping after initial_patience evaluations
        """
        if eval_every_steps == -1:
            eval_every_steps = round(
                len(train_dataset) / (4 * int(batch_size) * gradient_accumulation_steps)
            )
        compound_signature_input_dim = list(train_dataset.drugs_signatures.values())[
            0
        ].shape
        disease_signature_input_dim = list(train_dataset.diseases_signatures.values())[
            0
        ].shape[0]

        logger.info(
            "compound_signature_input_dim: %s, "
            "disease_signature_input_dim: %s, "
            % (
                compound_signature_input_dim,
                disease_signature_input_dim,
            )
        )

        model = self.model_type(
            # compound_fingerprint_input_dim=2048,
            compound_signature_input_dim=compound_signature_input_dim,
            disease_signature_input_dim=disease_signature_input_dim,
            # cell_line_input_dim=cell_line_input_dim,
            # dosage_input_dim=dosage_input_dim,
            internal_layer_size=internal_layer_size,
        )
        model.train()
        model.to(device=self.device)

        optimizer = Adam(params=model.parameters(), lr=float(lr))
        if balance_batches:
            train_sampler = get_weighted_sampler(labels=train_dataset.labels)
        else:
            train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=int(batch_size),
            num_workers=0,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=int(batch_size) * 4, shuffle=False, num_workers=0
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        total_steps = num_epochs * len(train_dataloader) / gradient_accumulation_steps
        logger.info("total steps: %s" % total_steps)
        logger.info("gradient accumulation %s" % gradient_accumulation_steps)
        logger.info("warmup steps: %s", warmup_steps)
        logger.info("Evaluate every %s steps" % eval_every_steps)

        step = 0
        patience = initial_patience
        current_auc = -1
        for epoch in range(num_epochs):
            logger.info("Epoch %s" % epoch)
            avg_train_loss = 0
            i = 0
            for batch in tqdm.tqdm(train_dataloader):
                batch = {k: b.to(self.device) for k, b in batch.items()}
                probs = model(**{k: v for k, v in batch.items() if not k == "label"})

                loss = criterion(probs, batch["label"].float())
                loss = loss / gradient_accumulation_steps
                loss.backward()
                if (
                        i + 1
                ) % gradient_accumulation_steps == 0:  # Wait for several backward steps
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()  # Now we can do an optimizer step
                    model.zero_grad()  # Reset gradients tensors
                avg_train_loss += loss.item()
                i += 1
                step += 1

                if step % eval_every_steps == 0:
                    logger.info(
                        "training loss: %s" % (avg_train_loss / eval_every_steps)
                    )
                    avg_train_loss = 0
                    predictions = self.predict_on_dataloader(
                        model=model, val_dataloader=val_dataloader, criterion=None
                    )
                    y_true = val_dataset.labels
                    metrics = evaluate(y_true=y_true, y_pred=predictions, threshold=0.5)
                    logger.info(metrics)
                    auc = metrics["auc"]
                    if auc > current_auc:
                        logger.info(
                            "AUC improved from %s to %s, saving model"
                            % (current_auc, auc)
                        )
                        current_auc = auc
                        torch.save(model, f=self.model_path)
                        patience = initial_patience
                    else:
                        patience -= 1
                    if patience <= 0:
                        logger.info(
                            "No improvement in AUC for %s evaluations, exiting"
                            % initial_patience
                        )
                        break
            if patience <= 0:
                break

    def predict_on_dataloader(
            self, model: Module, val_dataloader: DataLoader, criterion
    ):
        """
        Predict on a dataloader with model
        :param criterion: is a loss function used to calculate the loss on the dev set
        """
        preds = []
        avg_val_loss = 0
        with torch.no_grad():
            sigmoid = torch.nn.Sigmoid()
            for batch in tqdm.tqdm(val_dataloader):
                batch = {k: b.to(self.device) for k, b in batch.items()}
                probs = model(**{k: v for k, v in batch.items() if not k == "label"})
                if criterion:
                    loss = criterion(probs, batch["label"].squeeze().float())
                    avg_val_loss += loss.item()
                probs = sigmoid(probs)
                probs = probs.cpu().detach().tolist()
                for prob in probs:
                    preds.append(prob)
            if criterion:
                logger.info("val loss:%s" % (avg_val_loss / len(val_dataloader)))
        return preds

    def predict(
            self, test_dataset: OmicsDataset, batch_size: int, multi_gpu: bool = False
    ):
        """
        Predict on test dataset
        """
        model = torch.load(self.model_path)
        if multi_gpu:
            model = DataParallel(model)
        model.eval()
        model.to(self.device)
        test_dataloader = DataLoader(
            test_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
        )
        return self.predict_on_dataloader(
            model=model, val_dataloader=test_dataloader, criterion=None
        )

    def evaluate(self, test_dataset: OmicsDataset, batch_size: int):
        """
        Evaluate on a test dataset
        """
        model = torch.load(self.model_path)
        model.eval()
        model.to(self.device)
        logger.info("Results in test_set")
        test_dataloader = DataLoader(
            test_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
        )
        predictions = self.predict_on_dataloader(
            model=model, val_dataloader=test_dataloader, criterion=None
        )
        y_true = test_dataset.labels
        metrics = evaluate(y_true=y_true, y_pred=predictions)
        logger.info(metrics)
        return predictions

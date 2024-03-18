from utils.data_loaders import DialogSummaryDataset
from utils.bert import get_bert_base_uncased_model
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import lightning as L
from torch import nn
import torch
import os


class SequenceEncoder(nn.Module):
    """
    Sequence encoder using BERT
    The encoder takes a sequence of tokens and returns embeddings of the CLS token
    """
    def __init__(self):
        super(SequenceEncoder, self).__init__()
        self._bert = get_bert_base_uncased_model()

    @property
    def device(self):
        return self._bert.device

    def forward(self, x, mask=None):
        return self._bert(x, attention_mask=mask, output_hidden_states=True).hidden_states[-1][:, 0, :]


def cosine_distance(x, y):
    # calculate cosine distance between two tensors
    return 1 - F.cosine_similarity(x, y)


class LitTripletTraining(L.LightningModule):
    """
    PyTorch Lightning for Triplet Training
        - Anchor: Dialog text
        - Positive: Corresponding summary text
        - Negative: Hard negative selection for summary text
    """
    def __init__(self, train_dataset: DialogSummaryDataset, save_dir: str = 'models'):
        """
        Initialize the model
        :param train_dataset: training dataset - used to manipulate selected negative samples
        :param save_dir: directory to save model checkpoints
        """

        # store dataset and save directory
        self._train_ds = train_dataset
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = Path(save_dir)

        super().__init__()

        # store loss for plotting
        self._train_loss, self._val_loss = [], []

        # set loss function and class weights
        self.dialog_encoder = SequenceEncoder()
        self.summary_encoder = SequenceEncoder()

        # Initialize the loss function with the cosine distance
        self._criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5)

        # Set optimizer
        self._optimizer = AdamW
        self._learning_rate = 2e-5
        self._weight_decay = 1e-2

        # store predictions and labels for batch metrics calculation
        self._train_step_predictions, self._val_step_predictions = [], []

    def configure_optimizers(self):
        """
        Configure the optimizer
        :return: optimizer
        """
        # set optimizer arguments and initialize
        kwargs = {'lr': self._learning_rate, 'weight_decay': self._weight_decay}
        optimizer = self._optimizer(self.parameters(), **kwargs)
        return optimizer

    def accuracy(self, anchor, pos, neg):
        """
        Calculate accuracy
        :param anchor: anchor embeddings
        :param pos: positive embeddings
        :param neg: negative embeddings
        :return: accuracy
        """
        # calculate distances of pairs
        pos_distance = cosine_distance(anchor, pos)
        neg_distance = cosine_distance(anchor, neg)
        return (pos_distance < neg_distance).float()

    def on_train_epoch_start(self):
        # update embeddings according to the current epoch
        self._train_ds.set_emb(self.dialog_encoder, self.summary_encoder)

    def training_step(self, batch, batch_idx):
        """
        Training step
        :param batch: batch of data
        :param batch_idx: batch index
        :return: loss
        """

        # unpack batch and apply model
        anchor_tokens, anchor_attention_mask, pos_tokens, pos_attention_mask, neg_tokens, neg_attention_mask = batch
        anchor_emb = self.dialog_encoder(anchor_tokens, anchor_attention_mask)
        pos_emb = self.summary_encoder(pos_tokens, pos_attention_mask)
        neg_emb = self.summary_encoder(neg_tokens, neg_attention_mask)

        loss = self._criterion(anchor_emb, pos_emb, neg_emb)

        self._train_step_predictions.append(self.accuracy(anchor_emb, pos_emb, neg_emb))
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        :param batch: batch of data
        :param batch_idx: batch index
        :return: loss
        """
        # unpack batch and apply model
        anchor_tokens, anchor_attention_mask, pos_tokens, pos_attention_mask, neg_tokens, neg_attention_mask = batch
        anchor_emb = self.dialog_encoder(anchor_tokens, anchor_attention_mask)
        pos_emb = self.summary_encoder(pos_tokens, pos_attention_mask)
        neg_emb = self.summary_encoder(neg_tokens, neg_attention_mask)

        loss = self._criterion(anchor_emb, pos_emb, neg_emb)

        self._val_step_predictions.append(self.accuracy(anchor_emb, pos_emb, neg_emb))
        self.log('val_loss', loss, on_epoch=True)  # Log loss for the entire validation set
        return loss

    def on_train_epoch_end(self):
        """
        Calculate and log training metrics
        """

        # calculate metrics
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        accuracy = torch.cat(self._train_step_predictions).float().mean().item()

        # log metrics
        print(f'\n\nEpoch [{self.current_epoch + 1}/{self.trainer.max_epochs}] - '
              f'Training Loss: {train_loss:.4f}', f'Accuracy: {accuracy:.4f}')
        self._train_loss.append(train_loss)

    def on_validation_epoch_end(self):
        """
        Calculate and log validation metrics
        """

        # calculate metrics
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        accuracy = torch.cat(self._val_step_predictions).float().mean().item()

        # log metrics
        print(f'\nValidation Loss: {val_loss:.4f}', f'Accuracy: {accuracy:.4f}')
        self._val_loss.append(val_loss)

        torch.save(self.dialog_encoder.state_dict(), self._save_dir / 'checkpoint_dialog_encoder.pth')
        torch.save(self.summary_encoder.state_dict(), self._save_dir / 'checkpoint_summary_encoder.pth')
        torch.cuda.empty_cache()


def plot_loss(train_loss, val_loss, filename=None):
    """
    Plot loss over epochs
    :param train_loss: array of train loss - with shape (n_epochs,)
    :param val_loss: array of val loss - with shape (n_epochs,)
    :param filename: filename to save the plot, if None - show plot
    :return:
    """
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet-Loss")
    plt.title("Loss over epochs")
    plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


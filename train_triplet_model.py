from utils.data_loaders import DialogSummaryDataset, dialog_summary_collate_fn
from utils.triplet_model import LitTripletTraining, plot_loss
from torch.utils.data import DataLoader
from pathlib import Path
import lightning as L
import torch
import os

SAVE_DIR = 'models'
BATCH_SIZE = 16
NUM_WORKERS = 8
EPOCHS = 20
DEVICE = 'mps'


if __name__ == '__main__':
    # reference data
    dialog_csv_train = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_dialogues_train.csv'
    dialog_csv_test = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_dialogues_test.csv'
    summary_csv_train = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_summaries_train.csv'
    summary_csv_test = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_summaries_test.csv'

    ds_train = DialogSummaryDataset(dialog_csv_train, summary_csv_train)
    ds_test = DialogSummaryDataset(dialog_csv_test, summary_csv_test)

    dl_train = DataLoader(ds_train,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=dialog_summary_collate_fn,
                          num_workers=NUM_WORKERS,
                          persistent_workers=True)
    dl_val = DataLoader(ds_test, batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=dialog_summary_collate_fn,
                        num_workers=NUM_WORKERS,
                        persistent_workers=True)

    # load model
    save_dir = Path(SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)

    model = LitTripletTraining(train_dataset=ds_train, save_dir=save_dir.as_posix())

    # train model
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator=DEVICE) # , precision=16
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # plot loss
    plot_loss(model._train_loss, model._val_loss, filename=save_dir / 'loss.png')

    # save state dict
    dialog_encoder = model.dialog_encoder
    summary_encoder = model.summary_encoder

    torch.save(dialog_encoder.state_dict(), save_dir / 'dialog_encoder.pth')
    torch.save(summary_encoder.state_dict(), save_dir / 'summary_encoder.pth')
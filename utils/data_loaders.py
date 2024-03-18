import random
from utils.bert import get_bert_base_uncased_tokenizer, preprocess_text
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import re
import torch

tqdm.pandas()

# unisex names


def load_dialog(dialog_csv: str, sliding_window=False, window_size=10, stride=5) -> pd.DataFrame:
    """
    Load dialog data
    :param dialog_csv: dialog csv file
    :return: dialog data frame
    """
    df_dialog = pd.read_csv(dialog_csv)
    df_dialog.rename(columns={'id': 'instance_id', 'dialogue': 'text'}, inplace=True)

    # sliding window on rows
    if sliding_window:
        df_dialog['text'] = df_dialog['text'].progress_apply(lambda x: ['\n'.join(x.split("\n")[i:i + window_size])
                                                                        for i in range(0, len(x.split("\n")), stride)])
        df_dialog = df_dialog.explode('text')
    return df_dialog


def load_summary(summary_csv: str) -> pd.DataFrame:
    """
    Load summary data
    :param summary_csv: summary csv file
    :return: summary data frame
    """
    df_summary = pd.read_csv(summary_csv)
    if 'dialog_id' in df_summary.columns:
        df_summary.rename(columns={'dialog_id': 'instance_id'}, inplace=True)
    else:
        df_summary["instance_id"] = df_summary.index
    df_summary.rename(columns={'summary_piece': 'text'}, inplace=True)
    return df_summary


class TextDataset(Dataset):
    """
    Dataset class for the action enrichment task
    """

    def __init__(self, data_df: pd.DataFrame, tokenizer=None):
        """
        Dataset for text data
        each instance is a text sentence (or paragraph)

        :param data_df: data frame with columns: instance_id, text
        :param tokenizer: tokenizer
        """

        # set mode and tokenizer
        self._data_df = data_df
        self._tokenizer = get_bert_base_uncased_tokenizer() if tokenizer is None else tokenizer

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx):
        # get tokens and labels
        inst_id, text = self._data_df.iloc[idx][['instance_id', 'text']]

        # convert tokens to ids and prepare for model
        text = preprocess_text(text)
        pos_summary_tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
        pos_summary_tokens = self._tokenizer.prepare_for_model(pos_summary_tokens, return_tensors='pt')['input_ids']

        return inst_id, text, pos_summary_tokens


def text_collate_fn(batch: list) -> tuple:
    """
    Collate function for the action enrichment task
    :param batch: batch of data
    :return: Tuple of tokens, phrase_token_idx, phrase, labels
    """
    # extract batch based on mode (ActionDataset acts differently based on mode)
    inst_id, text, tokens = zip(*batch)

    # pad tokens with [PAD] - 0
    tokens, attention_mask = pad_sequence(tokens)

    # return data based on mode
    return inst_id, text, tokens, attention_mask


class DialogSummaryDataset(Dataset):
    """
    Dataset class for the action enrichment task
    """
    def __init__(self, dialog_csv, summary_csv, tokenizer=None, mask_names=True):
        """
        Dataset for dialog-summary data
        :param dialog_csv: dialog csv file
        :param summary_csv: summary csv file
        :param tokenizer: tokenizer (default: BERT base uncased)
        :param mask_names: if True, mask names in the dialog and summaries (default: True)
        """

        # set mode and tokenizer
        self._use_mask_names = mask_names
        self._tokenizer = get_bert_base_uncased_tokenizer() if tokenizer is None else tokenizer

        # load dialogs and summaries
        self._df_dialog = load_dialog(dialog_csv)
        self._df_summary = load_summary(summary_csv)
        # self._set_augmenter()

        # filter long dialogs
        self._filter_long_dialogs()
        self._dialog_embeddings = None
        self._summary_embeddings = None

    def _filter_long_dialogs(self, max_tokens=500):
        """
        Filter long dialogs
        :param max_tokens: instance with more than max_tokens will be dropped (default: 500)
        :return:
        """
        self._df_dialog['num_tokens'] = self._df_dialog['text'].progress_apply(
            lambda x: len(self._tokenizer.tokenize(x)))

        drop_instances = self._df_dialog.query(f'num_tokens > {max_tokens}')['instance_id'].tolist()
        self._df_summary = self._df_summary[~self._df_summary['instance_id'].isin(drop_instances)]
        self._df_dialog = self._df_dialog.query(f'num_tokens <= {max_tokens}')

    def _mask_names(self, dialog_txt, pos_txt, neg_summary_txt):
        if random.random() < 0.05:
            return dialog_txt, pos_txt, neg_summary_txt

        # get names
        names = set([r.split(':', 1)[0] for r in dialog_txt.strip().split("\n")])
        mask_f = lambda x: re.sub(rf"({'|'.join(names)})", "[MASK]", x, flags=re.IGNORECASE)

        return mask_f(dialog_txt), mask_f(pos_txt), mask_f(neg_summary_txt)

    def get_dialog_by_id(self, inst_id):
        return self._df_dialog.query(f"instance_id == '{inst_id}'").iloc[0]['text']

    def get_summary_by_id(self, inst_id, pos=None):
        if pos is None:
            return self._df_summary.query(f"instance_id == '{inst_id}'")

        return self._df_summary.query(f"instance_id == '{inst_id}' and position_index == {pos}")

    def set_emb(self, model_dialog, model_summary):
        self._dialog_embeddings = {
            inst_id: model_dialog(self._tokenizer.prepare_for_model(
                self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(dialog_txt)), return_tensors='pt')['input_ids'].unsqueeze(dim=0).to(model_dialog.device)).cpu().detach().numpy()
            for inst_id, dialog_txt in tqdm(self._df_dialog[['instance_id', 'text']].values, total=len(self._df_dialog), desc="Dialog Embeddings")
        }

        self._summary_embeddings = {
            inst_id: model_summary(self._tokenizer.prepare_for_model(
                self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(summary_txt)), return_tensors='pt')['input_ids'].unsqueeze(dim=0).to(model_dialog.device)).cpu().detach().numpy()
            for inst_id, summary_txt in tqdm(self._df_summary[['instance_id', 'text']].values, total=len(self._df_summary), desc="Summary Embeddings")
        }

    def __len__(self):
        return len(self._df_summary)

    def __getitem__(self, idx):
        pos_inst_id, pos_summary_txt = self._df_summary.iloc[idx][['instance_id', 'text']]
        dialog_txt = self._df_dialog.query(f"instance_id == '{pos_inst_id}'").iloc[0]['text']

        if self._dialog_embeddings is None or self._summary_embeddings is None:
            neg_inst_id, neg_summary_txt = self._df_summary.query(f"instance_id != '{pos_inst_id}'").sample(1).iloc[0][['instance_id', 'text']]
        else:
            dialog_emb = self._dialog_embeddings[pos_inst_id]
            # sample 10 random summaries != pos_summary
            sample_summary_keys = random.sample([k for k in self._summary_embeddings.keys() if k != pos_inst_id], 100)
            sample_cosine = {k: F.cosine_similarity(torch.tensor(dialog_emb), torch.tensor(self._summary_embeddings[k])).item()
                             for k in sample_summary_keys}
            neg_inst_id = max(sample_cosine, key=sample_cosine.get)
            neg_summary_txt = self._df_summary.query(f"instance_id == '{neg_inst_id}'").iloc[0]['text']

        # mask names
        if self._use_mask_names:
            dialog_txt, pos_summary_txt, neg_summary_txt = self._mask_names(dialog_txt, pos_summary_txt, neg_summary_txt)

        # preprocess text
        dialog_txt, pos_summary_txt, neg_summary_txt = map(lambda x: preprocess_text(x),
                                                           [dialog_txt, pos_summary_txt, neg_summary_txt])

        # tokenize
        pos_summary_tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(pos_summary_txt))
        pos_summary_tokens = self._tokenizer.prepare_for_model(pos_summary_tokens, return_tensors='pt')['input_ids']

        neg_summary_tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(neg_summary_txt))
        neg_summary_tokens = self._tokenizer.prepare_for_model(neg_summary_tokens, return_tensors='pt')['input_ids']

        dialog_tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(dialog_txt))
        dialog_tokens = self._tokenizer.prepare_for_model(dialog_tokens, return_tensors='pt')['input_ids']

        return dialog_tokens, pos_summary_tokens, neg_summary_tokens


def pad_sequence(tokens):
    # calculate max length
    max_len = max([len(t) for t in tokens])

    # create attention mask
    attention_mask = torch.zeros(len(tokens), max_len)
    for i, t in enumerate(tokens):
        attention_mask[i, :len(t)] = 1

    # pad tokens with [PAD] - 0
    tokens = [torch.cat((t, torch.zeros(max_len - len(t)).long())) for t in tokens]

    return torch.stack(tokens), attention_mask


def dialog_summary_collate_fn(batch: list) -> tuple:
    """
    Collate function for the action enrichment task
    :param batch: batch of data
    :return: Tuple of tokens, phrase_token_idx, phrase, labels
    """
    # extract batch based on mode (ActionDataset acts differently based on mode)
    anchor, pos, neg = zip(*batch)

    # pad summary-tokens with [PAD] - 0
    anchor_tokens, anchor_attention_mask = pad_sequence(anchor)
    pos_tokens, pos_attention_mask = pad_sequence(pos)
    neg_tokens, neg_attention_mask = pad_sequence(neg)

    return anchor_tokens, anchor_attention_mask, pos_tokens, pos_attention_mask, neg_tokens, neg_attention_mask

from utils.bert import get_bert_base_uncased_tokenizer, preprocess_text
from utils.triplet_model import SequenceEncoder
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


class Sorter:
    """
    Sorter class to find the order of the summaries in the dialogues
        - split the dialogues into N windows according to the number of summaries
        - calculate the cosine similarity between the dialog windows and the summaries
        - use a greedy approach to find the best order
    """

    def __init__(self, df_align, df_dialog, dialog_encoder_path, summary_encoder_path):
        """
        Initialize the Sorter class
        :param df_align: output of the Allocator class
        :param df_dialog: df of the dialogues
        :param dialog_encoder_path: dialog encoder state dict path
        :param summary_encoder_path: summary encoder state dict path
        """

        self._df_dialog = df_dialog
        self.df_align = df_align.groupby('pred_dialog_id').agg(list).reset_index()

        # load dialog Encoder
        self._dialog_encoder = SequenceEncoder()
        self._dialog_encoder.load_state_dict(torch.load(dialog_encoder_path))

        # load summary Encoder
        self._summary_encoder = SequenceEncoder()
        self._summary_encoder.load_state_dict(torch.load(summary_encoder_path))

        self._tokenizer = get_bert_base_uncased_tokenizer()

    def _preprocess_text(self, text):
        """
        Preprocess the text before encoding
        :param text: input text
        :return: tokens for encoding
        """
        text = preprocess_text(text)
        tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
        tokens = self._tokenizer.prepare_for_model(tokens, return_tensors='pt')['input_ids']

        return tokens.unsqueeze(0)

    def _sliding_window(self, text, n_windows):
        """
        Create a sliding window of the text with n_windows
        :param text: input text
        :param n_windows: Number of windows
        :return: windows
        """
        # split the text into words and calculate the window size
        lines = text.split(" ")
        win_size = int(len(lines) / n_windows)

        # get the start and end of each window
        start = [i * win_size for i in range(n_windows)]
        end = [len(lines) - (i * win_size) for i in range(n_windows)][::-1]

        # create the windows
        windows = [" ".join(lines[s:e])[:500] for s, e in zip(start, end)]
        return windows

    def find_order(self):
        """
        Find the order of the summaries in the dialogues
        :return: df with the order of the summaries
        """
        # get the order of the dialogues
        order = {}
        for i, instance in tqdm(self.df_align.iterrows(), total=len(self.df_align), desc="Finding order"):
            dialog_id, summaries = instance[['pred_dialog_id', 'summary']]
            dialog_text = self._df_dialog.query(f"instance_id == '{dialog_id}'").iloc[0]['text']

            # create a sliding window of the dialog
            windowed_dialog = self._sliding_window(dialog_text, n_windows=len(summaries))
            dialog_tokens = [self._dialog_encoder(self._preprocess_text(win)) for win in windowed_dialog]

            # get tokens for the summaries
            summary_tokens = [self._summary_encoder(self._preprocess_text(summary)) for summary in summaries]

            # create cosine similarity matrix
            distance_matrix = np.zeros((len(dialog_tokens), len(summary_tokens)))
            for i, dialog_emb in enumerate(dialog_tokens):
                for j, summary_emb in enumerate(summary_tokens):
                    distance_matrix[i, j] = F.cosine_similarity(dialog_emb, summary_emb).item()

            # greedy algorithm to find the best order
            best_order = []
            for i in range(len(summaries)):
                best_order.append(np.argmax(distance_matrix[:, i]))
                distance_matrix[best_order[-1], :] = -1

            order[dialog_id] = best_order

        order = pd.DataFrame({'pred_dialog_id': list(order.keys()), 'order': list(order.values())})
        self.df_align = self.df_align.merge(order, on='pred_dialog_id')
        return self.df_align

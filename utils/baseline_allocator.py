from utils.data_loaders import TextDataset, load_summary
from utils.index_utils import build_dialog_index
from utils.triplet_model import SequenceEncoder
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch


class Allocator:
    """
    Allocate summaries to dialogues
    """
    def __init__(self, dialog_csv, summary_csv, dialog_encoder_path, summary_encoder_path):
        """
        init
        :param dialog_csv: csv file with dialog data
        :param summary_csv: csv file with summary data
        :param dialog_encoder_path: path to dialog encoder model (state_dict)
        :param summary_encoder_path: path to summary encoder model (state_dict)
        """

        # set paths
        self._summary_csv = summary_csv

        # load dialog Encoder
        self._dialog_encoder = SequenceEncoder()
        self._dialog_encoder.load_state_dict(torch.load(dialog_encoder_path))

        # load summary Encoder
        self._summary_encoder = SequenceEncoder()
        self._summary_encoder.load_state_dict(torch.load(summary_encoder_path))

        # load dialog index
        self._index_dialog, self._id_mapping_dialog = build_dialog_index(dialog_csv, self._dialog_encoder)

    def _allocate_single_summary(self, tokens_summary, top_k=5):
        """
        Allocate a single summary to the dialogues
        :param tokens_summary: tokens for the summary
        :param top_k: return top-k similar dialogues
        :return:
        """
        # get the embeddings for the summary and normalize them
        emb_summary = self._summary_encoder(tokens_summary.unsqueeze(0)).cpu().detach().numpy()
        emb_summary = emb_summary / np.linalg.norm(emb_summary, axis=1)[:, np.newaxis]

        # search for the top-k similar dialogues
        D, I = self._index_dialog.search(emb_summary, top_k)
        D = D[0]
        I = np.asarray([self._id_mapping_dialog[i] for i in I[0]])
        return D, I

    def align_all(self, gt_data=False):
        """
        Allocate all summaries to dialogues by best match
        :param gt_data: if True input df is assumed to contain ground truth data (meaning instance-id is dialog_id)
        :return: aligned data frame
        """

        # load summary data
        df_summary = load_summary(self._summary_csv)
        if gt_data:
            df_summary['instance_id'] = df_summary['instance_id'] + "#" + df_summary['position_index'].astype(str)
        ds_summary = TextDataset(df_summary)
        summary_to_dialog = []

        for instance_id, text, tokens in tqdm(ds_summary, total=len(ds_summary), desc="Allocating summaries to dialogues"):
            D, I = self._allocate_single_summary(tokens)
            max_instance_id = I[0]

            # create prediction
            pred = {
                "summary": text,
                "dialog_sim": D[0],
                "pred_dialog_id": max_instance_id,
            }

            # add ground truth data
            if gt_data:
                instance_id, position_index = instance_id.split("#")
                pred['dialog_gt'] = instance_id
                pred['position_index_gt'] = int(position_index)
            else:
                pred['instance_id'] = instance_id
            summary_to_dialog.append(pred)

        df_align = pd.DataFrame(summary_to_dialog)
        return df_align


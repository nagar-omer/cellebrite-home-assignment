from utils.data_loaders import load_summary, TextDataset
from utils.index_utils import build_dialog_index
from utils.triplet_model import SequenceEncoder
from tqdm.auto import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import torch


class MaxFlowAllocator:
    def __init__(self, dialog_csv, summary_csv, dialog_encoder_path, summary_encoder_path):
        self._summary_csv = summary_csv

        # load dialog Encoder
        self._dialog_encoder = SequenceEncoder()
        self._dialog_encoder.load_state_dict(torch.load(dialog_encoder_path))

        # load summary Encoder
        self._summary_encoder = SequenceEncoder()
        self._summary_encoder.load_state_dict(torch.load(summary_encoder_path))

        # load dialog index
        self._index_dialog, self._id_mapping_dialog = build_dialog_index(dialog_csv, self._dialog_encoder)

        self._gnx = None

    def _allocate_single_summary(self, tokens_summary, top_k=5):
        # get the embeddings for the summary and normalize them
        emb_summary = self._summary_encoder(tokens_summary.unsqueeze(0)).cpu().detach().numpy()
        emb_summary = emb_summary / np.linalg.norm(emb_summary, axis=1)[:, np.newaxis]

        # search for the top-k similar dialogues
        D, I = self._index_dialog.search(emb_summary, top_k)
        D = D[0]
        I = np.asarray([self._id_mapping_dialog[i] for i in I[0]])
        return D, I

    def align_all(self, gt_data=False):
        # create Di-graph with source and target nodes
        self._gnx = nx.DiGraph()
        self._gnx.add_node('source')
        self._gnx.add_node('target')
        self._gnx.add_edges_from([(f"dialog_{i}", 'target', {'capacity': 3}) for i in self._id_mapping_dialog.values()])

        # load summary data
        df_summary = load_summary(self._summary_csv)
        if gt_data:
            df_summary['instance_id'] = df_summary['instance_id'] + "#" + df_summary['position_index'].astype(str)
        ds_summary = TextDataset(df_summary)

        for instance_id, text, tokens in tqdm(ds_summary, total=len(ds_summary), desc="Allocating summaries to dialogues"):
            D, I = self._allocate_single_summary(tokens)
            self._gnx.add_edges_from([(f"summary_{instance_id}", f"dialog_{i}", {'capacity': D[j]}) for j, i in enumerate(I)])
            self._gnx.add_edge('source', f"summary_{instance_id}", capacity=max(D))

        flow_value, flow_dict = nx.maximum_flow(self._gnx, "source", "target")
        summary_to_dialog = []
        for k, v in flow_dict.items():
            if 'summary_' in k:
                k = k.replace('summary_', '')
                v = max(v, key=v.get).replace('dialog_', '')

                # create prediction
                pred = {
                    "pred_dialog_id": v,
                    'summary': df_summary.query(f"instance_id == '{k}'").iloc[0]['text']
                }
                if gt_data:
                    instance_id, position_index = k.split("#")
                    pred['dialog_gt'] = instance_id
                    pred['position_index_gt'] = int(position_index)
                else:
                    pred['instance_id'] = k
                summary_to_dialog.append(pred)

        df_align = pd.DataFrame(summary_to_dialog)
        return df_align

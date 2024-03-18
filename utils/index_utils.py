from utils.data_loaders import load_dialog, TextDataset, text_collate_fn, load_summary
from torch.utils.data import DataLoader
from faiss import IndexFlatIP
from torch.nn import Module
from tqdm.auto import tqdm
from typing import Tuple
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def add_to_index(index: IndexFlatIP, embeddings: np.ndarray):
    """
    normalize embeddings and add to the index (faiss)
    :param index:
    :param embeddings:
    :return:
    """
    # Normalize the vectors to unit length for cosine similarity
    embeddings = embeddings.cpu().detach().numpy()
    unit_vectors = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Add vectors to the index
    index.add(unit_vectors)
    return index


def create_index(model: Module, dl: DataLoader, d=768) -> Tuple[IndexFlatIP, dict]:
    """
    Get embeddings for the dataset
    :param model: model
    :param dl: data loader
    :param d: embedding dimension
    :return: index, id_mapping
    """

    index = IndexFlatIP(d)
    id_mapping = {}
    for i, (inst_ids, texts, tokens, attention_mask) in tqdm(enumerate(dl), total=len(dl), desc="Building index"):

        # get token embeddings and extract CLS token
        with torch.no_grad():
            output = model(tokens, attention_mask)

        # update id-mapping {sequential_id: instance_id}
        max_key = max(id_mapping.keys()) if len(id_mapping) > 0 else -1
        id_mapping.update(dict(zip(range(max_key + 1, max_key + len(inst_ids) + 1), inst_ids)))

        # update index
        add_to_index(index, output)

    return index, id_mapping


def build_dialog_index(csv_path: str, model: Module):
    """
    Build index for dialog data
    :param csv_path: path to dialog csv
    :param model: model to use for encoding
    :return: index_dialog, id_mapping_dialog
    """
    df_dialog = load_dialog(csv_path, sliding_window=True)
    ds = TextDataset(df_dialog)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=text_collate_fn)
    index_dialog, id_mapping_dialog = create_index(model, dl, d=768)

    return index_dialog, id_mapping_dialog


def build_summary_index(csv_path: str, model: Module):
    """
    Build index for summary data
    :param csv_path: path to summary csv
    :param model: model to use for encoding
    :return: index_summary, id_mapping_summary
    """
    # check if the index and id_mapping files exist

    df_summary = load_summary(csv_path)
    ds = TextDataset(df_summary, tokenizer=None)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=text_collate_fn)
    index_summary, id_mapping_summary = create_index(model, dl, d=768)

    return index_summary, id_mapping_summary
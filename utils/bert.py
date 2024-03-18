from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import logging
import string
import os
import re

logger = logging.getLogger(__name__)

# set download root for models
DOWNLOAD_ROOT = Path(__file__).resolve().parent.parent / "cache" / "models"
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)


def get_bert_base_uncased_tokenizer(cache_dir: Path = DOWNLOAD_ROOT):
    """
    Get bert tokenizer
    :return: tokenizer
    """
    cache_dir = cache_dir / "bert_uncased"
    logger.info("Getting bert tokenizer")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        return tokenizer

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None


def get_bert_base_uncased_model(cache_dir: Path = DOWNLOAD_ROOT):
    """
    Get bert uncased model
    :return: tokenizer, PT model
    """

    cache_dir = cache_dir / "bert_uncased"
    logger.info("Getting bert uncased model")
    try:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              cache_dir=cache_dir,
                                                              output_hidden_states=True)
        return model
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None


def preprocess_text(text: str, tokenizer=None) -> str:
    """
    Preprocess text data.
    :param text: text to preprocess
    :param tokenizer: tokenizer

    - convert to lower case
    - remove non-ascii characters, punctuation and extra spaces.
    - tokenize the text
    """
    # convert to lower case
    text = text.lower()

    # remove non-ascii characters
    text = text.encode("ascii", "ignore").decode()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    if tokenizer:
        return tokenizer.tokenize(text)
    return text
# from utils.bert import get_bert_base_uncased_tokenizer
from utils.bert import get_bert_base_uncased_tokenizer
from utils.data_loaders import load_dialog, load_summary
import nlpaug.augmenter.word as naw
from random import random
from tqdm.auto import tqdm
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tqdm.pandas()


class TextAugmenter:
    """
    Class for text augmentation
    """
    def __init__(self):
        self.syn_aug, self.bt_aug, self.insert_aug, self.delete_aug = self._get_augmenter()

    def _get_augmenter(self) -> tuple:
        # Synonym Replacement
        syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
        # Back Translation
        bt_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en',
        )
        # Insertion
        insert_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", aug_p=0.1)
        # Deletion
        delete_aug = naw.RandomWordAug(action="delete", aug_p=0.1)
        return syn_aug, bt_aug, insert_aug, delete_aug

    def augment(self, text):
        if random() > 0.1:
            return self.bt_aug.augment(text)

        text = self.syn_aug.augment(text)
        text = self.insert_aug.augment(text)
        text = self.delete_aug.augment(text)
        return text


def augment_row(row, augmenter, k=10):
    new_text_col = [row['text']]
    new_text_col.extend([augmenter.augment(row['text']) for _ in range(k)])
    row['text'] = new_text_col
    return row


def count_tokens(text, tokenizer):
    return len(tokenizer.tokenize(text))


def augment(df, k=10):
    augmenter = TextAugmenter()
    print('Augmenting data...')
    return df.progress_apply(augment_row, axis=1, args=(augmenter, k))


def filter_long_dialogs(df, max_tokens=512):
    tokenizer = get_bert_base_uncased_tokenizer()
    print('Filtering long dialogs...')
    df['num_tokens'] = df['dialogue'].progress_apply(count_tokens, args=(tokenizer,))
    return df.query(f'num_tokens <= {max_tokens}')


def train_test_split(dialog_csv, summary_csv, test_size=0.2):
    """
    Split the data into train and test
        - filter long dialogs
        - split the data into train and test by dialog id
    :param dialog_csv: csv file with dialog data
    :param summary_csv: csv file with summary data
    :param test_size: test size
    """
    # names for the new files
    test_dialog_csv = dialog_csv.replace('.csv', '_test.csv')
    test_summary_csv = summary_csv.replace('.csv', '_test.csv')

    train_dialog_csv = dialog_csv.replace('.csv', '_train.csv')
    train_summary_csv = summary_csv.replace('.csv', '_train.csv')

    # read the data
    df_dialog = pd.read_csv(dialog_csv)
    df_summary = pd.read_csv(summary_csv)

    # filter long dialogs
    df_dialog = filter_long_dialogs(df_dialog)
    df_summary = df_summary[df_summary['dialog_id'].isin(df_dialog['id'])]

    # split the data
    df_dialog_test = df_dialog.sample(frac=test_size, random_state=42)
    df_summary_test = df_summary[df_summary['dialog_id'].isin(df_dialog_test['id'])]

    df_dialog_train = df_dialog[~df_dialog['id'].isin(df_dialog_test['id'])]
    df_summary_train = df_summary[df_summary['dialog_id'].isin(df_dialog_train['id'])]

    # save the data
    df_dialog_test.to_csv(test_dialog_csv, index=False)
    df_summary_test.to_csv(test_summary_csv, index=False)
    df_dialog_train.to_csv(train_dialog_csv, index=False)
    df_summary_train.to_csv(train_summary_csv, index=False)


if __name__ == '__main__':
    dialog_csv = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_dialogues.csv'
    summary_csv = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_summaries.csv'
    train_test_split(dialog_csv, summary_csv, test_size=0.2)
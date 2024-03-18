from utils.baseline_allocator import Allocator
from utils.data_loaders import load_dialog
from utils.sorter import Sorter

DIALOG_ENCODER_PATH = 'models/dialog_encoder.pth'
SUMMARY_ENCODER_PATH = 'models/summary_encoder.pth'


def count_inversions(arr):
    inversions = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions


def calc_inversion_accuracy(gt, pred):
    if len(gt) == 1:
        return 1 if gt == pred else 0
    max_inversions = len(gt) * (len(gt) - 1) // 2  # Maximum inversions when the prediction is in reverse order

    gt_inversions = count_inversions(gt)
    pred_inversions = count_inversions(pred)

    accuracy = 1 - (abs(gt_inversions - pred_inversions) / max_inversions)
    return max(accuracy, 0)  # Ensure accuracy is between 0 and 1


def performance_ref_data(ref_dialog_csv, ref_summary_csv):
    allocator = Allocator(ref_dialog_csv, ref_summary_csv,
                                 dialog_encoder_path=DIALOG_ENCODER_PATH,
                                 summary_encoder_path=SUMMARY_ENCODER_PATH)
    df_align = allocator.align_all(gt_data=True)

    sorter = Sorter(df_align, load_dialog(ref_dialog_csv),
                    dialog_encoder_path=DIALOG_ENCODER_PATH,
                    summary_encoder_path=SUMMARY_ENCODER_PATH)
    df_align = sorter.find_order()
    apply_f = lambda x: calc_inversion_accuracy(x['order'], list(range(len(x['order']))))
    df_align['inversion_accuracy'] = df_align.apply(apply_f, axis=1)


def predict(dialog_csv, summary_csv, out_csv):
    allocator = Allocator(dialog_csv, summary_csv,
                                 dialog_encoder_path=DIALOG_ENCODER_PATH,
                                 summary_encoder_path=SUMMARY_ENCODER_PATH)
    df_align = allocator.align_all(gt_data=False)

    sorter = Sorter(df_align, load_dialog(dialog_csv),
                    dialog_encoder_path=DIALOG_ENCODER_PATH,
                    summary_encoder_path=SUMMARY_ENCODER_PATH)
    df_align = sorter.find_order()
    df_align = df_align.explode(['summary', 'instance_id', 'order'])
    df_align = df_align[['pred_dialog_id', 'summary', 'order']].rename(columns={
        'pred_dialog_id': 'dialog_id',
        'summary': 'summary_piece',
        'order': 'position_index'
    })
    df_align.to_csv(out_csv, index=False)


if __name__ == '__main__':
    dialog_csv_ = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_dialogues_test.csv'
    summary_csv_ = '/Users/omernagar/Documents/Projects/cellebrite-home-assignment/data/reference_summaries_test.csv'
    # performance_ref_data(dialog_csv, summary_csv)

    dialog_csv_ = 'data/dialogues.csv'
    summary_csv_ = 'data/summary_pieces.csv'
    out_csv_ = 'data/aligned_summaries.csv'
    predict(dialog_csv_, summary_csv_, out_csv_)
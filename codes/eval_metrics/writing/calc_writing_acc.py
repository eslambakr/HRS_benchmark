import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm
from nltk import edit_distance
import fastwer
from cer import calculate_cer


def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_obj = []
    for sample in gt_data:
        gt_obj.append(sample['chatgpt_out'][1:-1])  # [1:-1] to remove the double quotes

    return gt_obj


def load_pred(pkl_pth, iter_idx):
    with open(pkl_pth, 'rb') as f:
        pred_data = pickle.load(f)

    # Split pred data based on iter_idx:
    pred_txt = {}
    for sample in pred_data:
        iter_id = int(sample['img_name'].split(".png")[0].split("_")[-1])
        if not (iter_id in pred_txt.keys()):
            pred_txt[iter_id] = []

        # Merge detected text segments together:
        detected_txt = ""
        for word in sample['detected_txt']:
            detected_txt = detected_txt + " " + word.lower()
        sample['detected_txt'] = detected_txt
        pred_txt[iter_id].append(sample)

    return pred_txt[iter_idx]


def cal_acc(gt_txt, pred_txt):
    """
    gt_txt: list of strings
    pred_txt: list of Dict
    """
    # Calculate the Acc:
    ned, cer, wer = 0, 0, 0
    for i in range(len(gt_txt)):
        pred, gt = pred_txt[i]['detected_txt'], gt_txt[i]
        distance = edit_distance(gt, pred)
        # Follow ICDAR 2019 definition of N.E.D.
        ned += distance / max(len(pred), len(gt))
        # Obtain Sentence-Level Character Error Rate (CER)
        cer += calculate_cer(pred.split(" "), gt.split(" "))
        # Obtain Sentence-Level Word Error Rate (WER)
        wer += fastwer.score_sent(pred, gt)

    return [(100*ned)/len(gt_txt), (100*cer)/len(gt_txt), (100*wer)/len(gt_txt)]


if __name__ == "__main__":
    # Load GT:
    gt_txt = load_gt(csv_pth='../../prompt_gen/synthetic_writing_prompts.csv')
    iter_num = 5
    ned, cer, wer = [], [], []
    for iter_idx in range(iter_num):
        # Load Predictions:
        pred_txt = load_pred(pkl_pth='mmocr_pred_writing.pkl', iter_idx=iter_idx)
        # Calculate the counting Accuracy:
        writing_acc = cal_acc(gt_txt, pred_txt)
        ned.append(writing_acc[0])
        cer.append(writing_acc[1])
        wer.append(writing_acc[2])
        print("NED ", iter_idx, ": ", writing_acc[0], "%")
        print("CER ", iter_idx, ": ", writing_acc[1], "%")
        print("WER ", iter_idx, ": ", writing_acc[2], "%")
    print("----------------------------")
    print("NED: ", sum(ned)/len(ned), "%")
    print("CER: ", sum(cer)/len(cer), "%")
    print("WER: ", sum(wer)/len(wer), "%")
    print("Done!")

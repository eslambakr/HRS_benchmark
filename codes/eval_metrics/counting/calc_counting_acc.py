import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm
import sys


def convert_csv_2_txt():
    meta_f = open("meta_emotions.txt", "w")
    synthetic_f = open("synthetic_emotions.txt", "w")
    data = pd.read_csv("/home/eslam/Downloads/synthetic_emotion_prompts.csv").to_dict('records')
    for sample in data:
        meta_f.write(sample['meta_prompt']+"\n")
        synthetic_f.write(sample['synthetic_prompt']+"\n")
    meta_f.close()
    synthetic_f.close()


def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_obj = []
    for sample in gt_data:
        temp_dict = {sample['obj1']: sample['n1']}
        if sample['n2'] > 0:
            temp_dict[sample['obj2']] = sample['n2']
        gt_obj.append(temp_dict)

    return gt_obj


def load_pred(pkl_pth, iter_idx):
    with open(pkl_pth, 'rb') as f:
        pred_data = pickle.load(f)

    pred_data = pred_data[iter_idx]
    # Keep class info only. Discard box coordinates info:
    pred_objs = {}
    for img_id, v in pred_data.items():
        temp_dict = {}
        for obj_id, v2 in v.items():
            temp_lst = []
            for item in v2:
                temp_lst.append(item[-1].lower())
            temp_dict[obj_id] = temp_lst

        pred_objs[img_id] = temp_dict

    return pred_objs


def cal_acc(gt_objs, pred_objs, level):
    # Calculate the Acc:
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for img_id, sample in enumerate(gt_objs):
        img_id += (level * int(len(gt_objs) / 3))
        for obj_name, gt_num in sample.items():
            pred_num = 0
            for k, pred_obj in pred_objs[img_id].items():
                if obj_name in pred_obj:
                    pred_num += 1
            true_pos += min(gt_num, pred_num)
            false_pos = false_pos + max((pred_num-gt_num), 0)
            false_neg = false_neg + max((gt_num-pred_num), 0)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return [precision, recall]


if __name__ == "__main__":
    in_pkl = sys.argv[1]
    gt_csv = sys.argv[2]
    iter_num = int(sys.argv[3])  # e.g., 1
    # Load GT:
    gt_objs = load_gt(csv_pth=gt_csv)
    precisions, recalls, f1 = [], [], []
    precisions_per_level = {0: [], 1: [], 2: []}
    recalls_per_level = {0: [], 1: [], 2: []}
    f1_per_level = {0: [], 1: [], 2: []}
    for iter_idx in range(iter_num):
        for level in range(3):
            mul = int(len(gt_objs) / 3)
            # Load Predictions:
            pred_objs = load_pred(pkl_pth=in_pkl, iter_idx=iter_idx)
            # Calculate the counting Accuracy:
            precision, recall = cal_acc(gt_objs[level*mul:(level+1)*mul], pred_objs, level=level)
            precision *= 100
            recall *= 100
            precisions.append(precision)
            recalls.append(recall)
            f1.append((2*precision*recall)/(precision+recall))
            print("precision ", iter_idx, ": ", precision, "%")
            print("recall ", iter_idx, ": ", recall, "%")
            print("F1 Score ", iter_idx, ": ", f1[-1], "%")
            # Per level:
            precisions_per_level[level].append(precision)
            recalls_per_level[level].append(recall)
            f1_per_level[level].append((2 * precision * recall) / (precision + recall))

    for level in range(3):
        print("----------------------------")
        if level == 0:
            print("   Easy level Results   ")
        elif level == 1:
            print("   Medium level Results   ")
        elif level == 2:
            print("   Hard level Results   ")
        print("precision: ", (sum(precisions_per_level[level]) / len(precisions_per_level[level])), "%")
        print("recall: ", (sum(recalls_per_level[level]) / len(recalls_per_level[level])), "%")
        print("F1 Score : ", (sum(f1_per_level[level]) / len(f1_per_level[level])), "%")
    print("----------------------------")
    print("   Average level Results   ")
    print("precision: ", (sum(precisions)/len(precisions)), "%")
    print("recall: ", (sum(recalls)/len(recalls)), "%")
    print("F1 Score : ", (sum(f1)/len(f1)), "%")
    print("Done!")

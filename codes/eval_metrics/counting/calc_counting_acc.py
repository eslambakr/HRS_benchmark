import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm


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


def cal_acc(gt_objs, pred_objs):
    # Calculate the Acc:
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for img_id, sample in enumerate(gt_objs):
        for obj_name, gt_num in sample.items():
            pred_num = 0
            for k, pred_obj in pred_objs[img_id].items():
                if obj_name in pred_obj:
                    pred_num += 1
            true_pos += min(gt_num, pred_num)
            false_pos = false_pos + max((pred_num-gt_num), 0)
            false_neg = false_neg + max((gt_num-pred_num), 0)

    precision = true_pos / (true_pos + false_pos)  # top, center, side, back
    # trash can, laundry hamper, bathroom stall door, paper towel dispenser, coffee table, kitchen cabinet,
    # office chair, copier, kitchen cabinets, end table, kitchen counter, file cabinet, oven
    recall = true_pos / (true_pos + false_neg)
    return [precision, recall]


if __name__ == "__main__":
    # Load GT:
    gt_objs = load_gt(csv_pth='../../prompt_gen/synthetic_counting_prompts_15.csv')
    iter_num = 5
    precisions, recalls = [], []
    for iter_idx in range(iter_num):
        # Load Predictions:
        pred_objs = load_pred(pkl_pth='unidet_pred_synthetic_counting.pkl', iter_idx=iter_idx)
        # Calculate the counting Accuracy:
        counting_acc = cal_acc(gt_objs, pred_objs)
        precisions.append(counting_acc[0])
        recalls.append(counting_acc[1])
        print("precision ", iter_idx, ": ", counting_acc[0] * 100, "%")
        print("recall ", iter_idx, ": ", counting_acc[1] * 100, "%")
    print("----------------------------")
    print("precision: ", (sum(precisions)/len(precisions)) * 100, "%")
    print("recall: ", (sum(recalls)/len(recalls)) * 100, "%")
    print("Done!")

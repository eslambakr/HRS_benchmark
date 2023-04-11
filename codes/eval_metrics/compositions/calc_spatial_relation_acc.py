import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm
import os, sys


def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_list = []
    for sample in gt_data:
        # Objects:
        objs = [sample['obj1'], sample['obj2']]
        for i in range(3, 5):
            if type(sample['obj' + str(i)]) is str:  # check if there is an object
                objs.append(sample['obj' + str(i)])

        # Relations:
        relations = [sample['rel1']]
        if type(sample['rel2']) is str:  # check if there is a second relation
            relations.append(sample['rel2'])

        gt_list.append({"objs": objs, "relations": relations})

    return gt_list


def load_pred(pkl_pth, iter_idx):
    with open(pkl_pth, 'rb') as f:
        pred_data = pickle.load(f)

    pred_data = pred_data[iter_idx]
    # Keep class info only. Discard box coordinates info:
    pred_objs = {}
    for img_id, v in pred_data.items():
        temp_dict = {}
        for obj_id, v2 in v.items():
            item = v2[0]  # remove duplicate objects
            temp_dict[obj_id] = {"cls": item[-1]}
            # convert coordinates to float instead of str:
            cords = [float(cord) for cord in item[:4]]
            temp_dict[obj_id]["cords"] = cords  # (xmin, ymin, xmax, ymax)  # origin = top left

        pred_objs[img_id] = temp_dict

    return pred_objs


def _check_right(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmax1 > xmax2:
        return True
    else:
        return False


def _check_left(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmin1 < xmin2:
        return True
    else:
        return False


def _check_above(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymin1 < ymin2) or (ymax1 < ymax2):
        return True
    else:
        return False


def _check_below(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymax1 > ymax2) or (ymin1 > ymin2):
        return True
    else:
        return False


def _check_between(obj_1, obj_2, obj_3):
    """
    Check if obj1 in between of obj2 and obj3
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
        obj_3: coordinates of object 3 (xmin, ymin, xmax, ymax)
    """
    # check horizontal dimension:
    if _check_right(obj_1, obj_2) and _check_left(obj_1, obj_3):
        return True
    elif _check_left(obj_1, obj_2) and _check_right(obj_1, obj_3):
        return True
    # check vertical dimension:
    elif _check_below(obj_1, obj_2) and _check_above(obj_1, obj_3):
        return True
    elif _check_above(obj_1, obj_2) and _check_below(obj_1, obj_3):
        return True
    else:
        return False


def _sort_pred_obj(pred_objs, gt_objs):
    """
    Sorting the predicted objects based on the GT objects.
    pred_objs: dict of pred objs. key --> obj_id. val --> cls and cords.
    gt_objs: list of gt cls names.
    """
    sorted_pred_objs = {}
    for key, pred_obj in pred_objs.items():
        if pred_obj['cls'] in gt_objs:
            sorted_pred_objs[gt_objs.index(pred_obj['cls'])] = pred_obj
    return sorted_pred_objs


def cal_acc(gt_objs, pred_objs, level):
    above_spatial_words = ["on", "above", "over"]
    below_spatial_words = ["below", "beneath", "under"]
    relative_relations = ["between", "among", "in the middle of"]
    true_count = 0
    for img_id, sample in enumerate(gt_objs):
        img_id += (level * int(len(gt_objs) / 3))
        miss_flag = False
        # Get the whole predicted classes in this image:
        pred_cls = [pred_objs[img_id][obj_id]['cls'] for obj_id in pred_objs[img_id].keys()]

        # Check whether the image contains the correct classes or not:
        for obj_cls in sample['objs']:
            if obj_cls in pred_cls:
                continue
            else:
                miss_flag = True
                break
        if miss_flag:
            continue

        # Sorting the predicted objects based on the GT objects
        sorted_pred_objs = _sort_pred_obj(pred_objs[img_id], sample['objs'])

        # Determine the hardness level based on the number of objects:
        if len(sample['objs']) == 2:
            # Easy level:
            if sample['relations'][0] == "on the right of":
                if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] == "on the left of":
                if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] in above_spatial_words:
                if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] in below_spatial_words:
                if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
        elif len(sample['objs']) == 3:
            # Medium level:
            if len(sample['relations']) == 2:  # normal relation
                # Check first relation:
                if sample['relations'][0] == "on the right of":
                    if not _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] == "on the left of":
                    if not _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] in above_spatial_words:
                    if not _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] in below_spatial_words:
                    if not _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue

                # Check second relation:
                if sample['relations'][1] == "on the right of":
                    if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] == "on the left of":
                    if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] in above_spatial_words:
                    if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] in below_spatial_words:
                    if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1

            else:  # between relation
                if _check_between(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords'],
                                  sorted_pred_objs[2]['cords']):
                    true_count += 1

        elif len(sample['objs']) == 4:
            # Hard level:
            if len(sample['relations']) == 2:  # normal relation
                # Check first relation:
                if sample['relations'][0] == "on the right of":
                    if not (_check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and
                            (_check_right(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']))):
                        continue
                elif sample['relations'][0] == "on the left of":
                    if not (_check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and
                            (_check_left(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']))):
                        continue
                elif sample['relations'][0] in above_spatial_words:
                    if not (_check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and
                            (_check_above(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']))):
                        continue
                elif sample['relations'][0] in below_spatial_words:
                    if not (_check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and
                            (_check_below(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']))):
                        continue

                # Check second relation:
                if sample['relations'][1] == "on the right of":
                    if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[4]['cords']) and \
                            _check_right(sorted_pred_objs[1]['cords'], sorted_pred_objs[4]['cords']):
                        true_count += 1
                elif sample['relations'][1] == "on the left of":
                    if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[4]['cords']) and \
                            _check_left(sorted_pred_objs[1]['cords'], sorted_pred_objs[4]['cords']):
                        true_count += 1
                elif sample['relations'][1] in above_spatial_words:
                    if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[4]['cords']) and \
                            _check_above(sorted_pred_objs[1]['cords'], sorted_pred_objs[4]['cords']):
                        true_count += 1
                elif sample['relations'][1] in below_spatial_words:
                    if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[4]['cords']) and \
                            _check_below(sorted_pred_objs[1]['cords'], sorted_pred_objs[4]['cords']):
                        true_count += 1

            else:  # between relation
                if _check_between(sorted_pred_objs[0]['cords'],
                                  sorted_pred_objs[2]['cords'],
                                  sorted_pred_objs[3]['cords']) \
                        and _check_between(sorted_pred_objs[1]['cords'],
                                           sorted_pred_objs[2]['cords'],
                                           sorted_pred_objs[3]['cords']):
                    true_count += 1
        else:
            raise Exception("Sorry, number of objects should be between 1-4")

    acc = 100 * (true_count / len(gt_objs))
    return acc


if __name__ == "__main__":
    in_pkl = sys.argv[1]
    gt_csv = sys.argv[2]
    iter_num = int(sys.argv[3])  # 3
    # Load GT:
    gt_data = load_gt(csv_pth=gt_csv)
    avg_acc = []
    acc_per_level = {0: [], 1: [], 2: []}
    for iter_idx in range(iter_num):
        for level in range(3):
            mul = int(len(gt_data) / 3)
            # Load Predictions:
            pred_data = load_pred(pkl_pth=in_pkl, iter_idx=iter_idx)
            # Calculate the counting Accuracy:
            acc = cal_acc(gt_data[level * mul:(level + 1) * mul], pred_data, level=level)
            avg_acc.append(acc)
            print("Accuracy ", iter_idx, ": ", acc, "%")
            # Per level:
            acc_per_level[level].append(acc)

    for level in range(3):
        print("----------------------------")
        if level == 0:
            print("   Easy level Results   ")
        elif level == 1:
            print("   Medium level Results   ")
        elif level == 2:
            print("   Hard level Results   ")
        print("Accuracy: ", (sum(acc_per_level[level]) / len(acc_per_level[level])), "%")
    print("----------------------------")
    print("   Average level Results   ")
    print("Averaged Accuracy: ", (sum(avg_acc) / len(avg_acc)), "%")

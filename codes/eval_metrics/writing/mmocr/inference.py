import os
import glob
from tqdm import tqdm
import pickle
from mmocr.ocr import MMOCR

# Load models into memory
det_model_name = "TextSnake"
rec_model_name = "SAR"
ocr = MMOCR(det=det_model_name, pred_score_thr=0.55, recog=rec_model_name)

# Define in & out dirs:
input_dir = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/sd_v1/writing_15/"
base_out_dir = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/metrics/writing/"
out_dir = base_out_dir + "mmocr_" + det_model_name + "_" + rec_model_name
isExist = os.path.exists(out_dir)
if not isExist:
   os.makedirs(out_dir)

# Inference
output_lst_dict = []
imgs_names = os.listdir(input_dir)
imgs_names = sorted(imgs_names)
for img_name in tqdm(imgs_names):
    img_pth = os.path.join(input_dir, img_name)
    results = ocr.readtext(img_pth, print_result=False, show=False, img_out_dir=out_dir)
    # print("img:", img_name, "-->", results[0]['det_scores'])
    if results['det_scores']:
        rec_texts, rec_scores = [], []
        for i in range(len(results['rec_scores'])):
            if results['rec_scores'][i] >= 0.60:
                rec_texts.append(results['rec_texts'][i])
                rec_scores.append(results['rec_scores'][i])
                # print("img:", img_name, "-->", results['rec_texts'][i], " -->", results['rec_scores'][i])
        output_lst_dict.append({"img_name": img_name, "detected_txt": rec_texts, "detected_scores": rec_scores})
    else:
        output_lst_dict.append({"img_name": img_name, "detected_txt": [], "detected_scores": []})

# Save results in pkl file:
with open('../mmocr_pred_writing.pkl', 'wb') as f:
    pickle.dump(output_lst_dict, f)

print("Done!")

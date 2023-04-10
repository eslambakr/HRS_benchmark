import os
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np
import math
from math import factorial
from chatGPT_inference import save_lst_strings_to_txt


def get_missed_images(path):
    dir_list = os.listdir(path)
    print("-------------")
    for i in range(3000):
        for j in range(5):
            if not (str(i).zfill(5)+"_"+str(j).zfill(2)+".png" in dir_list):
                print(i)
                break


def clean_fid_for_cogview_2():
    """
    clip long sentences
    """
    f = open("/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/codes/prompt_gen/misspelling_prompts/spell_3.txt", "r")
    saving_f = open("/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/codes/prompt_gen/misspelling_prompts/spell_3_cog.txt", "w")
    lines = f.readlines()
    print(len(lines))
    for line in tqdm(lines):
        words = line.split(" ")
        new_line = ' '.join(words[:26])
        if not ("\n" in new_line):
            new_line += "\n"
        saving_f.write(new_line)
    saving_f.close()


def gen_fairness_styles_csv(chatgpt_csv):
    fairness_styles = ["animation", "real", "sketch", "sunny", "cloudy", "rainy", "black and white"]
    chatgpt_out = pd.read_csv(chatgpt_csv).to_dict('records')
    prompts = []
    for style in fairness_styles:
        for sample in chatgpt_out:
            prompt = "a {style} scene about {chatgpt_out}".format(style=style, chatgpt_out=sample["synthetic_prompt"])
            prompts.append(prompt)

    save_lst_strings_to_txt(saving_txt="fairness_styles" + "_prompts.txt", lst_str=prompts)


def _merge_3hardness_folder_in_one(parent_folder, in_folders, saving_dir, base_count):
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    for idx, folder in enumerate(in_folders):
        path = os.path.join(parent_folder, folder)
        imgs = os.listdir(path)
        cur_base = idx * base_count
        for img in imgs:
            new_img_name = str(int(img.split("_")[0]) + cur_base).zfill(5)+"_"+img.split("_")[1]
            shutil.copyfile(os.path.join(path, img), os.path.join(saving_dir, new_img_name))


def gen_fairness_gender_csv(chatgpt_csv):
    chatgpt_out = pd.read_csv(chatgpt_csv).to_dict('records')
    male_prompts, female_prompts = [], []
    for idx, sample in enumerate(chatgpt_out):
        if (0 <= idx < 250) or (350 <= idx < 600) or (750 <= idx < 1000):
            male_prompts.append(sample["synthetic_prompt"].replace("person", "man").replace("people", "men"))
            female_prompts.append(sample["synthetic_prompt"].replace("person", "woman").replace("people", "women"))

    prompts = male_prompts + female_prompts
    save_lst_strings_to_txt(saving_txt="fairness_gender" + "_prompts.txt", lst_str=prompts)


def sample_images(org_dir, saving_dir):
    np.random.seed(2023)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    selected_imgs = np.random.choice(a=np.arange(0, 1000), size=100, replace=False)
    for img_name in selected_imgs:
        new_img_name = str(img_name).zfill(5)+"_00.png"
        shutil.copyfile(os.path.join(org_dir, new_img_name), os.path.join(saving_dir, new_img_name))


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def check_valid_char(txt_file_pth):
    f = open(txt_file_pth, "r")
    prompts = f.readlines()
    for i, prompt in enumerate(prompts):
        if isEnglish(prompt):
            continue
        else:
            print("Line number is ", i+1, "--> the text is ", prompt)


def fairness_score():
    inp = [0.3329, 0.3225, 0.3189, 0.3223, 0.3238, 0.3228, 0.3228]
    n = 7
    k = 2
    score = 0
    C = factorial(n) / (factorial(k) * factorial(n - k))
    for i in range(len(inp)):
        for j in range(i + 1, len(inp)):
            score = score + ((100 * abs(inp[i] - inp[j])) / (C * max(inp[i], inp[j])))
    print(score / C)


def sample_quantative_results(models_dir):
    selected_imgs = {
        "fidelity_original": [87, 91, 104, 120, 142, 150, 163, 195, 273],
        "synthetic_counting": [2090, 2113, 2123, 2129, 2145, 2152, 2171, 2199, 2220],
        "writing": [825, 856, 947, 1066, 1072, 1100, 1112, 1135, 1162],
        "emotions": [45, 100, 118, 176, 87, 167, 196, 223, 258],
        "spatial": [28, 45, 150, 509, 597, 600, 651, 672, 705],
        "size": [50, 100, 150, 180, 200, 240, 280, 350, 400],
        "colors": [110, 170, 333, 355, 400, 435, 470, 490, 500],
        "actions": [20, 50, 90, 120, 150, 180, 210, 250, 351],
        "creativity_hard": [11, 48, 68, 77, 101, 109, 145, 156, 550],
        "bias": [86, 129, 159, 178, 213, 233, 251, 292, 333]
    }
    models = ["cogview2", "dalle_mini", "dalle_v2", "glide", "minidalle", "paella", "sd_v1", "sd_v2"]
    skills = ["fidelity_original", "synthetic_counting", "writing", "emotions",
              "spatial", "size", "colors", "actions", "creativity_hard", "bias"]
    for i in range(2, 11):
        for skill in skills:
            new_img_name = str(selected_imgs[skill][i-2]).zfill(5) + "_00.png"
            for model in models:
                imgs_pth = os.path.join(os.path.join(models_dir, model), skill)
                saving_dir = os.path.join(os.path.join(os.path.join(models_dir, "qualitative_results"),
                                                       "ex"+str(i)), skill)
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)
                saving_dir = os.path.join(saving_dir, model+".png")
                shutil.copyfile(os.path.join(imgs_pth, new_img_name),
                                saving_dir)


if __name__ == '__main__':
    # get_missed_images("/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/cogview2/vanilla_counting")

    # clean_fid_for_cogview_2()

    # gen_fairness_styles_csv(chatgpt_csv="./fairness/styles/synthetic_fairness_styles_prompts.csv")

    """
    parent_folder = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/dalle_mini/toeslam/dalle-mini-new/"
    _merge_3hardness_folder_in_one(parent_folder=parent_folder, in_folders=["styles_1", "styles_2"],
                                   saving_dir=os.path.join(parent_folder, "merged"), base_count=800)
    """

    # gen_fairness_gender_csv(chatgpt_csv="/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/codes/prompt_gen/bias/gender_race_age/synthetic_bias_prompts.csv")

    """
    sample_images(org_dir="/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/sd_v2/synthetic_bias/",
                  saving_dir="/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/human_eval/bias/sd_v2")
    """

    # check_valid_char("/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/codes/prompt_gen/compositional_skill/spatial/meta_spatial_relation_prompts.txt")

    models_dir = "/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out"
    sample_quantative_results(models_dir)

import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm


def save_csv(list_of_dict, save_file):
    keys = list_of_dict[0].keys()

    with open(save_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dict)


def convert_csv_2_txt_consistency():
    keys_of_interest = ["original_prompts", "edited_prompts_1", "edited_prompts_2", "edited_prompts_3"]
    human_data = pd.read_csv("../../prompt_gen/ver_2.csv").to_dict('records')
    human_data = human_data[:500]
    human_data = [{k: v for k, v in d.items() if k != 'Unnamed: 0'} for d in human_data]
    human_data = [{k: v for k, v in d.items() if k != 'keep'} for d in human_data]
    synthetic_data = pd.read_csv("../../prompt_gen/chatgpt_3.csv").to_dict('records')
    synthetic_data = synthetic_data[:500]
    data = synthetic_data + human_data
    save_csv(data, "../../prompt_gen/fidelity_prompts_and_T5.csv")
    for key in keys_of_interest:
        txt_file = open("../../prompt_gen/fidelity_" + key + ".txt", "w")
        for sample in data:
            txt_file.write(sample[key]+"\n")
        txt_file.close()


def convert_csv_2_txt_misspelling():
    keys_of_interest = ["original_prompt", "e1", "e2", "e3"]
    human_data = pd.read_csv("../../prompt_gen/human_spell.csv").to_dict('records')
    human_data = human_data[:500]
    synthetic_data = pd.read_csv("../../prompt_gen/generated_spell.csv").to_dict('records')
    synthetic_data = synthetic_data[:500]
    data = synthetic_data + human_data
    save_csv(data, "../../prompt_gen/misspelling_prompts.csv")
    for key in keys_of_interest:
        txt_file = open("../../prompt_gen/misspelling_" + key + ".txt", "w")
        for sample in data:
            txt_file.write(sample[key]+"\n")
        txt_file.close()


if __name__ == "__main__":
    convert_csv_2_txt_consistency()
    convert_csv_2_txt_misspelling()


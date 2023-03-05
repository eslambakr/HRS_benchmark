"""
Generate the real text prompt which will be fed to text-2-image models based on the fed meta-prompt.
"""
from meta_prompt_gen import MetaPromptGen
import openai
import datetime
from tqdm import tqdm
import csv
import sys


def run_chatgpt(model, temp, meta_prompt, max_tokens):
    # Define the parameters for the text generation
    completions = openai.Completion.create(engine=model, prompt=meta_prompt, max_tokens=max_tokens, n=1, stop=None,
                                           temperature=temp)
    gen_prompt = completions.choices[0].text.strip().lower()
    # Print the generated text
    print("The meta prompt is --> ", meta_prompt)
    print("ChatGPT output is --> ", gen_prompt)
    return gen_prompt


def save_lst_strings_to_txt(saving_txt, lst_str):
    file = open(saving_txt, 'w')
    for item in lst_str:
        file.write(item + "\n")
    file.close()


def save_prompts_in_csv(lst, saving_name):
    # Save output in csv:
    keys = lst[0].keys()
    with open(saving_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(lst)


def wait_one_n_mins(n_mins=1):
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=n_mins)
    while True:
        if datetime.datetime.now() >= endTime:
            break


if __name__ == '__main__':
    # Set your API key
    openai.api_key = sys.argv[1]

    meta_prompt_gen = MetaPromptGen(ann_path="../../data/metrics/det/lvis_v1/lvis_v1_train.json",
                                    label_space_path="../eval_metrics/detection/UniDet-master/datasets/label_spaces/learned_mAP+M.json",
                                    )
    num_prompts = 216
    skill = "fairness_styles"
    generated_lst_dict = []
    for i in tqdm(range(num_prompts)):
        meta_prompt_dict = meta_prompt_gen.gen_meta_prompts(level_id=int(i // (num_prompts / 3)), skill=skill)
        template = meta_prompt_dict["meta_prompt"]
        if int(i // (num_prompts / 3)) == 2 and skill == "fidelity":
            max_tokens = 50
        else:
            max_tokens = 40

        # Handle openAI timeout:
        chatgpt_out = None
        while chatgpt_out is None:
            if (skill == "spatial_relation") or (skill == "size_comp") or (skill == "color_comp"):
                # we will not use ChatGPT for this skill.
                break
            try:
                chatgpt_out = run_chatgpt(model="text-davinci-003", temp=0.5, meta_prompt=template,
                                          max_tokens=max_tokens)
            except:
                print("OpenAI server out! Will try again Don't worry :D")
                pass

        if skill == "writing":
            if int(i // (num_prompts / 3)) == 2:  # Hard level
                final_prompt = "a real scene of {place} with a sign written on it {chatgpt_out}".format(
                    place=meta_prompt_gen.select_rand_place(), chatgpt_out=chatgpt_out)
            else:  # Easy & Medium levels
                final_prompt = "a sign written on it {chatgpt_out}".format(chatgpt_out=chatgpt_out)
            meta_prompt_dict.update({"chatgpt_out": chatgpt_out})
        else:
            final_prompt = chatgpt_out
        meta_prompt_dict.update({"synthetic_prompt": final_prompt})
        generated_lst_dict.append(meta_prompt_dict)

        if (i % 20 == 0) and (i != 0) and chatgpt_out:
            wait_one_n_mins(n_mins=1)  # wait one minute to not exceed the openai limits

    generated_dict_lst = {k: [dic[k] for dic in generated_lst_dict] for k in generated_lst_dict[0]}

    # Saving:
    save_prompts_in_csv(lst=generated_lst_dict, saving_name="synthetic_" + skill + "_prompts.csv")
    if skill == "counting":
        save_lst_strings_to_txt(saving_txt="vanilla_" + skill + "_prompts.txt",
                                lst_str=generated_dict_lst['vanilla_prompt'])
    save_lst_strings_to_txt(saving_txt="meta_" + skill + "_prompts.txt", lst_str=generated_dict_lst['meta_prompt'])
    save_lst_strings_to_txt(saving_txt="synthetic_" + skill + "_prompts.txt",
                            lst_str=generated_dict_lst['synthetic_prompt'])

    print("Done")

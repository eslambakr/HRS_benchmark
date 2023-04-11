"""
"../../prompt_gen/emotion_skill/meta_emotion_prompts.txt"
"../../../data/t2i_out/dalle_v2/emotion_skill"
3
"Bearer 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"

"../../prompt_gen/misspelling_prompts/spell_1.txt"
"../../../data/t2i_out/dalle_v2/spell_1"
3
"Bearer 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"
"""
import requests
import json
import os, sys
import datetime
from tqdm import tqdm


def wait_one_n_mins(n_mins=1):
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=n_mins)
    while True:
        if datetime.datetime.now() >= endTime:
            break


def read_prompts_from_txt(txt_pth):
    f = open(txt_pth, "r")
    return f.readlines()


def save_img_from_url(img_url, saving_pth):
    while True:
        try:
            response = requests.get(img_url)
            if response.status_code:
                fp = open(saving_pth, 'wb')
                fp.write(response.content)
                fp.close()
                break
        except:
            print("OpenAI saving image server out! Will try again Don't worry :D")
            pass


prompt_file = sys.argv[1]
output_dir = sys.argv[2]
n_iter = sys.argv[3]
openai_key = sys.argv[4]

total_num_img = 0
# Define the API endpoint URL
api_url = "https://api.openai.com/v1/images/generations"
# Set up the API authentication headers
headers = {"Content-Type": "application/json", "Authorization": openai_key}

prompts = read_prompts_from_txt(prompt_file)
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir)

# Create the text tokens to feed to the model.
#
start_id = 0
for idx in tqdm(range(start_id, len(prompts))):  # len(prompts)
    # Handle the case where we wanna resume the generation.
    prompt = prompts[idx]
    for n in range(int(n_iter)):
        saving_path = os.path.join(output_dir, str(idx).zfill(5) + '_' + str(n).zfill(2)) + '.png'
        # Define the image generation parameters
        data = {"model": "image-alpha-001", "prompt": prompt, "num_images": 1, "size": "512x512",
                "response_format": "url"}
        # Send the API request
        # Handle openAI timeout:
        response_data = None
        safety_issue = False
        while True:
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(data))
                # print(response.text)
                response_data = json.loads(response.text)
                if "data" in response_data.keys():
                    break
                elif "safety system" in response.text:
                    safety_issue = True
                    print("Image number: ", idx, " is rejected due to safety concerns")
                    break
                else:
                    if (total_num_img % 50 == 0) and (total_num_img != 0):
                        wait_one_n_mins(n_mins=1)  # wait one minute to not exceed the openai limits
            except:
                print("OpenAI server out! Will try again Don't worry :D")
                pass

        # Extract the generated image URL from the response
        # print(response_data)
        if safety_issue:
            continue
        image_url = response_data["data"][0]["url"]
        save_img_from_url(image_url, saving_path)
        total_num_img += 1
        if (total_num_img % 50 == 0) and (total_num_img != 0):
            wait_one_n_mins(n_mins=1)  # wait one minute to not exceed the openai limits

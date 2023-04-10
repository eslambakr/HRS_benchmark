# from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="prompt file")
    parser.add_argument("--save_path", type=str, help="prompt file")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    prompts = df['prompt']

    prompts_imgs = []
    for prompt in tqdm(prompts, total=len(prompts)):
        item = defaultdict()
        client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=100)
        results = client.query(text=prompt)
        imgs = []
        for r in results:
            image_url = r['url']
            imgs.append(image_url)
            # sim.append(r['similarity'])
        item['prompt'] = prompt
        item['img_url'] = imgs
        prompts_imgs.append(item)
    with open(args.save_path, 'w') as f:
        json.dump(prompts_imgs, f)

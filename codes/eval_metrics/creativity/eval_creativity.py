import json
import torch
import clip
from PIL import Image
import os
import numpy as np
import pandas as pd

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from transformers import Blip2Processor, Blip2ForConditionalGeneration

import argparse


def compute_max(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="model name")
    parser.add_argument("--file_path", type=str, help="prompt file")
    parser.add_argument("--img_emb_path", type=str, help="NN training data embedding file")
    args = parser.parse_args()

    scorer_cider = Cider()
    bleu1 = Bleu(n=1)
    bleu2 = Bleu(n=2)
    bleu3 = Bleu(n=3)
    bleu4 = Bleu(n=4)
    tokenizer = PTBTokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bit8 = False
    dtype = {'load_in_8bit': True} if bit8 else {'torch_dtype': torch.float16}
    model_name = 'Salesforce/blip2-opt-2.7b-coco'
    processor = Blip2Processor.from_pretrained(model_name)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name, **dtype)
    blip2.to(device)

    model, preprocess = clip.load('ViT-B/32', device=device)

    df = pd.read_csv(args.file_path)

    prompts = df['prompt']
    image_paths = df['imgs']

    train_img_embs = torch.load(args.img_emb_path).to(device)

    img_sims, cider_scores, bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], [], [], []

    for idx, text_prompt in enumerate(prompts):
        gen_image_embd = []
        for image_path in enumerate(image_paths[idx]):
            image = Image.open(image_path).convert('RGB')
            gen_image_embd.append(preprocess(image).to(device))

            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                generated_id = blip2.generate(**inputs, do_sample=True, top_p=0.95, num_return_sequences=5)
                generated_texts = processor.batch_decode(generated_id, skip_special_tokens=True)
                candidates = []
                for generated_text in generated_texts:
                    candidates.append(generated_text.strip())
                cider_scores.append(compute_max(scorer_cider, text_prompt, candidates))
                bleu1_scores.append(compute_max(bleu1, text_prompt, candidates))
                bleu2_scores.append(compute_max(bleu2, text_prompt, candidates))
                bleu3_scores.append(compute_max(bleu3, text_prompt, candidates))
                bleu4_scores.append(compute_max(bleu4, text_prompt, candidates))

        gen_image_embd = torch.stack(gen_image_embd)
        train_image_embd = train_img_embs[idx]
        with torch.no_grad():
            gen_image_embd = model.encode_image(gen_image_embd).half()
        gen_image_embd /= gen_image_embd.norm(dim=-1, keepdim=True)
        train_image_embd /= train_image_embd.norm(dim=-1, keepdim=True)
        img_sim = (gen_image_embd @ train_image_embd.T).cpu()
        img_sim = torch.mean(img_sim)
        img_sims.append(img_sim)

deviation = 1.0 - (np.mean(img_sim) + 1.0) * 0.5
print(deviation, np.mean(cider_scores), np.mean(bleu1_scores), np.mean(bleu2_scores), np.mean(bleu3_scores),
      np.mean(bleu4_scores))

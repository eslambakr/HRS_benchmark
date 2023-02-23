"""
"../../prompt_gen/synthetic_writing_prompts.txt" "../../../data/t2i_out/minidalle/writing" 5
"""
from matplotlib import pyplot as plt
import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score
import numpy as np
import os, sys
from tqdm import tqdm
from PIL import Image


def read_prompts_from_txt(txt_pth):
    f = open(txt_pth, "r")
    return f.readlines()


prompt_file = sys.argv[1]
output_dir = sys.argv[2]
n_iter = sys.argv[3]
device = 'cuda:0'
set_seed(0)
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

prompts = read_prompts_from_txt(prompt_file)
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir)

# Sampling
for idx, prompt in tqdm(enumerate(prompts)):
    images = model.sampling(prompt=prompt,
                            top_k=256,  # It is recommended that top_k is set lower than 256.
                            top_p=None,
                            softmax_temperature=1.0,
                            num_candidates=int(n_iter),
                            device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # Save images
    for n in range(int(n_iter)):
        saving_path = os.path.join(output_dir, str(idx).zfill(5) + '_' + str(n).zfill(2)) + '.png'
        im = Image.fromarray((images[n]*255).astype(np.uint8))
        im.save(saving_path)

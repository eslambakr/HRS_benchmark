"""
"../../prompt_gen/synthetic_writing_prompts.txt" "../../../data/t2i_out/paella/writing" 5
"""
import os
import time
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from modules import DenoiseUNet
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
from einops import rearrange
import requests
import json
import os, sys
import datetime
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def showmask(mask):
    plt.axis("off")
    plt.imshow(torch.cat([
        torch.cat([i for i in mask[0:1].cpu()], dim=-1),
    ], dim=-2).cpu())
    plt.show()


def showimages(imgs, **kwargs):
    plt.figure(figsize=(kwargs.get("width", 32), kwargs.get("height", 32)))
    plt.axis("off")
    plt.imshow(torch.cat([
        torch.cat([i for i in imgs], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def saveimages(imgs, path, **kwargs):
    torchvision.utils.save_image(imgs, path, **kwargs)


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def sample(model, c, x=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0],
           typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11,
           renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T + 1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        preds = []
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1 - mask) * x
        init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            # x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1 - mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i + 1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i + 1], random_x=prev_x)
                else:  # 'rand'
                    x, _ = model.add_noise(x, r_range[i + 1])
            preds.append(x.detach())
    return preds


vqmodel = get_vae().to(device)
vqmodel.eval().requires_grad_(False)

clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
clip_model = clip_model.to(device).eval().requires_grad_(False)

clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711)),
])

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
])


def encode(x):
    return vqmodel.model.encode((2 * x - 1))[-1][-1]


def decode(img_seq, shape=(32, 32)):
    img_seq = img_seq.view(img_seq.shape[0], -1)
    b, n = img_seq.shape
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=vqmodel.num_tokens).float()
    z = (one_hot_indices @ vqmodel.model.quantize.embed.weight)
    z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
    img = vqmodel.model.decode(z)
    img = (img.clamp(-1., 1.) + 1) * 0.5
    return img


def read_prompts_from_txt(txt_pth):
    f = open(txt_pth, "r")
    return f.readlines()


state_dict = torch.load("../../../weights/t2i/paella/model_600000.pt", map_location=device)
# state_dict = torch.load("./models/f8_img_40000.pt", map_location=device)
model = DenoiseUNet(num_labels=8192).to(device)
model.load_state_dict(state_dict)
model.eval().requires_grad_()
print()

# ---------------------------------------
#           Text-Conditional
# ---------------------------------------
prompt_file = sys.argv[1]
output_dir = sys.argv[2]
n_iter = sys.argv[3]
batch_size = n_iter
mode = "text"
latent_shape = (32, 32)

prompts = read_prompts_from_txt(prompt_file)
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir)

# Create the text tokens to feed to the model.
total_num_img = 0
for idx, prompt in tqdm(enumerate(prompts)):
    tokenized_text = tokenizer.tokenize([prompt] * batch_size).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            clip_embeddings = clip_model.encode_text(tokenized_text)
            sampled = sample(model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1,
                             classifier_free_scale=5, renoise_steps=11, renoise_mode="start")
        sampled = decode(sampled[-1], latent_shape)

    for n in range(int(n_iter)):
        saving_path = os.path.join(output_dir, str(idx).zfill(5) + '_' + str(n).zfill(2)) + '.png'
        saveimages(sampled[n], saving_path, nrow=1)

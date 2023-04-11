# Prompts Generation:

You don't need to run the prompts generation codes as we already provide the generated prompts and can be downloaded from this [link](https://drive.google.com/drive/folders/1AlA259sXi-3ZJD7RFaL2bGDwJXLImJrx).

However, we provide also all the generation codes in this folder, where you can run it as follows:
## Creativity skill:
We adopted [LAION fast retrieval tool](https://github.com/rom1504/clip-retrieval) to retrieve training data (Nearest Neighbours) from LAION with text prompts.

```
python retrieve.py --file_path path/to/prompt --save_path path/to/prompt_with_img
```

And save the training data clip embedding with [CLIP](https://github.com/mlfoundations/open_clip)

```python
model, preprocess = clip.load('ViT-B/32', device=device)
image = Image.open(image_path).convert('RGB')
inputs = preprocess(images=image, return_tensors="pt").to(device, torch.float16)
with torch.no_grad():
	gen_image_embd = model.encode_image(inputs).half()
```


## Rest of skills:
Run our [prompt generation file](https://github.com/eslambakr/T2I_benchmark/blob/main/codes/prompt_gen/chatGPT_inference.py) 
with the desired arguments:
```
1) Your openai.api_key
2) The desired num of generated prompts  # e.g., 1000
3) The desired skill  # e.g., "bias"
```
For instance:
```
python chatGPT_inference.py [your openai_api_key] 3000 "counting"
```
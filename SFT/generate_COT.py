import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


prompts = {
        "orign": """
                Solve the following user instruction based on the screenshot through logical steps. Use the format "click ITEM" to provide a solution to execute the user's instructions, briefly describing the features and areas of the ITEM. User instruction: {}
            """,
        "constrains": """
                Solve the following user instruction based on the screenshot through three logical steps: 1. Screen information analysis: describe the screen information that may be involved based on the user's instructions. 2. Instruction reasoning: Based on the screen information, infer which items on the screen the user's instructions might control. 3. Solution generation: Use the format "click ITEM" to provide a solution to execute the user's instructions, briefly describing the features and areas of the ITEM. User instruction: {}
        """
    }

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, default="orign")
    parser.add_argument("--seed_file", type=str, default="uibert_sft")
    args = parser.parse_args()
    return args

def read_json(data_path):
	with open(data_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

args = get_parser()
reasoning_prompt = prompts.get(args.prompt_type)

uibert = read_json(f"/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/{args.seed_file}.json")

model_id = "/home/yuqi_zhou/PLMs/Align-DS-V"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to('cuda')

processor = AutoProcessor.from_pretrained(model_id)
for image_point in tqdm(uibert):
    img_file = image_point['img_filename']
    elements = image_point['elements']
    for element in elements:
        img_url = f"/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/UIBert/{img_file}"
        query = element['instruction']
        bbox = element['bbox']
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": reasoning_prompt.format(query)},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        raw_image = Image.open(img_url)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to('cuda', torch.float16)

        output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        cot = processor.decode(output[0], skip_special_tokens=True).split('<｜Assistant｜>')[1]
        element['cot'] = cot

with open(f'/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/{args.seed_file}_{args.prompt_type}_cot.json', 'w') as file:
    json.dump(uibert, file)

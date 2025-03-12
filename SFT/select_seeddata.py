
import os
model_dir="/home/yuqi_zhou/PLMs"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import ast
import torch
from PIL import Image, ImageDraw
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import argparse

import os
import json
from PIL import Image
from tqdm import tqdm
import re
def get_xy(output_text):
	numbers = re.findall(r'\d+', output_text[0])
	numbers = [int(i) for i in numbers]
	x, y = (numbers[0] + numbers[2])/2, (numbers[1] + numbers[3])/2
	return x, y
    
def is_point_inside_bbox(x, y, bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    return False

def read_json(data_path):
	with open(data_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

def get_parser():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	return args

def get_xy(output_text):
	coordinates = re.findall(r'\((\d+),(\d+)\)', output_text[0])
	coordinates = [[int(x), int(y)] for x, y in coordinates]
	x, y = (coordinates[0][0] + coordinates[1][0])/2, (coordinates[0][1] + coordinates[1][1])/2
	return x, y

model = Qwen2VLForConditionalGeneration.from_pretrained(
    f"{model_dir}/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

min_pixels = 256*28*28
max_pixels = 1344*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

args = get_parser()

data_path = '/home/yuqi_zhou/Documents/Grounding/dataset/{}/raw.json'
uibert = read_json(data_path.format('UIBERT'))
new_uibert = []

for image_point in tqdm(uibert):
    new_elements = []
    img_file = image_point['img_filename'].split('/')[1]
    elements = image_point['elements']
    for element in elements:
        img_url = f"/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/UIBert/{img_file}"
        query = element['instruction']
        bbox = element['bbox']
        _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},
                    {"type": "image", "image": img_url, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": query}
                ],
            }
        ]
        
        image = Image.open(img_url)
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        click_xy = ast.literal_eval(output_text)
        x, y = click_xy[0] * image.width, click_xy[1] * image.height
        if is_point_inside_bbox(x, y, bbox) is False:
            new_elements.append(element)
    if len(new_elements) > 0:
        new_uibert.append({
            'img_filename': img_file,
            'elements': new_elements
        })
    
    if len(new_uibert) == 1000:
        with open('/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/uibert_sft_1000.json', 'w') as file:
            json.dump(new_uibert, file)

with open('/home/yuqi_zhou/Documents/Grounding-R1/datasets/mobile_domain/uibert_sft.json', 'w') as file:
    json.dump(new_uibert, file)
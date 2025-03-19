import re
import ast
import torch
import json
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

prompt = "Grounding instruction is:{}. Help to locate and output its bbox coordinates using JSON format."

def is_point_inside_bbox(x, y, bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    return False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platfrom", type=str, default="mobile")
    parser.add_argument("--dataset", type=str, default="ScreenSpot-v2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_size", type=int, default=7)
    args = parser.parse_args()
    return args

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

args = get_parser()
data_path = f"/home/yuqi_zhou/Documents/Grounding/dataset/{args.dataset}/screenspot_{args.platfrom}_v2.json"
model_base_dir = "/home/yuqi_zhou/PLMs"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    f"{model_base_dir}/Qwen/Qwen2.5-VL-{args.model_size}B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(f"{model_base_dir}/Qwen/Qwen2.5-VL-{args.model_size}B-Instruct")
processor.tokenizer.padding_side = 'left'

with open(data_path, 'r') as file:
    examples = json.load(file)

right_cnts, wrong_cnts = 0, 0
for i in tqdm(range(0, len(examples), args.batch_size)):
    batch = examples[i:i + args.batch_size]
    img_urls = [f"/home/yuqi_zhou/Documents/Grounding/dataset/{args.dataset}/screenspotv2_image/{e['img_filename']}" for e in batch]
    queries = [e['instruction'] for e in batch]
    bboxes = [e['bbox'] for e in batch]

    messages = [
        [
            {
            "role": "system",
            "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_url},
                    {"type": "text", "text": prompt.format(query)},
                ],
            }
        ]
        for img_url, query in zip(img_urls, queries)
    ]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
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
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    bounding_boxes = [parse_json(output_text) for output_text in output_texts]
    for output_text, predicts, bbox in zip(output_texts, bounding_boxes, bboxes):
        print(bbox)
        print(predicts)
        try:
            parsed_expr = ast.literal_eval(predicts)
            bbox_2d = parsed_expr[0]['bbox_2d']
            if len(bbox_2d) == 4:
                x, y = (bbox_2d[0]+bbox_2d[2])/2, (bbox_2d[1]+bbox_2d[3])/2
            elif len(bbox_2d) == 2:
                x, y = bbox_2d[0], bbox_2d[1]
            else:
                continue
            if is_point_inside_bbox(x, y, bbox) is True:
                right_cnts += 1
            else:
                wrong_cnts += 1
        except:
            wrong_cnts += 1
            continue
    print(f"Success Rate: {right_cnts / (right_cnts + wrong_cnts)}\n")
    
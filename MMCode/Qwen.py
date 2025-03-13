
import os
import re
import json
import random
import base64
from io import BytesIO
import tempfile

from tqdm import tqdm
import numpy as np

from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def resize(image, base_width=None, base_height=None):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    if base_width:
        if base_width <= original_width:
            return image
        w_percent = (base_width / float(original_width))
        new_height = int((float(original_height) * float(w_percent)))
        new_size = (base_width, new_height)
    elif base_height:
        if base_height <= original_height:
            return image
        h_percent = (base_height / float(original_height))
        new_width = int((float(original_width) * float(h_percent)))
        new_size = (new_width, base_height)
    else:
        raise ValueError("Either base_width or base_height must be specified")

    # Resize the image
    resized_img = image.resize(new_size, Image.LANCZOS)
    return resized_img


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class Qwen:
    def __init__(self, path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(path)
    
    def generate(self, prompt, images, temperature=0.0):
        prompt = (
            "You are required to solve a programming problem. " 
            + "Please enclose your code inside a ```python``` block. " 
            + " Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" \
            + prompt
        )

        # Convert all images to base64
        base64_images = [convert_pil_image_to_base64(resize(image, base_height=480)) for image in images]

        interleaved_messages = []
        # Split the prompt and interleave text and images
        segments = re.split(r'!\[image\]\(.*?\)', prompt)
        for i, segment in enumerate(segments):
            # Text
            if len(segment) > 0:
                interleaved_messages.append({"type": "text", "text": segment})
            # Image
            if i < len(images):
                interleaved_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_images[i]}",
                    }
                })

        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a professional programming contester trying to solve algorithmic problems. The problems come with a description and some images, and you should write a Python solution."}
                ],
            },
            {
                "role": "user",
                "content": interleaved_messages
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids =self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return response

    def extract_code(self, response):
        pattern = r"```python(.*?)```"
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
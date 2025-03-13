import re
import torch
from PIL import Image
from typing import List
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class LLaVA:
    def __init__(self, model_path: str, model_base: str = None):
        # Disable initial torch operations
        disable_torch_init()
        
        # Load model components
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, self.model_name
        )
        self.device = self.model.device
        
        # Initialize conversation mode
        self._detect_conv_mode()

    def _detect_conv_mode(self):
        """Auto detect conversation mode based on model name"""
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def _build_prompt(self, problem_prompt: str, images: List[Image.Image]) -> str:
        """Construct LLaVA compatible prompt with image tokens"""
        # Replace image placeholders with tokens
        if DEFAULT_IMAGE_TOKEN in problem_prompt:
            if self.model.config.mm_use_im_start_end:
                image_token = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
            else:
                image_token = DEFAULT_IMAGE_TOKEN
            problem_prompt = problem_prompt.replace("![image]()", image_token)
        else:
            image_token = DEFAULT_IMAGE_TOKEN
            problem_prompt = f"{image_token}\n{problem_prompt}"

        # Initialize conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], problem_prompt)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()

    def generate(self, prompt: str, images: List[Image.Image], temperature: float = 0.2) -> str:
        # Process input images
        image_tensors = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.device, dtype=torch.float16)

        # Build full prompt with conversation template
        full_prompt = self._build_prompt(prompt, images)
        
        # Tokenize the prompt with image tokens
        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # Configure generation parameters
        generation_kwargs = {
            "do_sample": temperature > 1e-5,
            "temperature": temperature,
            "max_new_tokens": 1024,
            "images": image_tensors,
            "image_sizes": [img.size for img in images],  # Keep original image sizes
            "use_cache": True
        }

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                **generation_kwargs
            )

        # Decode and clean response
        response = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        # Remove duplicate assistant messages
        if response.startswith("ASSISTANT:"):
            response = response.split("ASSISTANT:")[-1].strip()
            
        return response

    def extract_code(self, response: str) -> str:
        """Extract Python code block from model response"""
        code_pattern = r"```python(.*?)```"
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            code = matches[0].strip()
            # Remove possible language markers in code
            return re.sub(r"^python\n", "", code)
        return response

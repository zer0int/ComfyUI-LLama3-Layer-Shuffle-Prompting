import os
import re
import torch
from torch import Tensor, nn
import time 
import copy
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import pipeline


class LLama3ShuffleBase:
    def __init__(self, load_from_file=False):
        super().__init__()
        load_from_file=load_from_file
        self.models = {}
        self.model = None
        self.original_model = None
        self.tokenizer = None

    def get_original_model(self, LLama3_model):
        if LLama3_model not in self.models:
            print(f"LLama-3 Shuffle: Switching model at: {time.time()}")
            del self.original_model
            self.original_model = None
            del self.model
            self.model = None
            torch.cuda.empty_cache()
                
        if self.load_from_file:
            if self.original_model is not None:
                del self.original_model
                self.original_model = None
                torch.cuda.empty_cache()

            if self.model is not None:
                del self.model
                self.model = None
                torch.cuda.empty_cache()
                print(f"LLama-3 Shuffle: File-load-always is enabled, deleted models at: {time.time()}")  
                
            self.models[LLama3_model] = LlamaForCausalLM.from_pretrained(LLama3_model)
            self.models[LLama3_model].eval()                   
            self.original_model = self.models[LLama3_model]
            print(f"LLama-3 Shuffle: Loaded LLama-3 from file at: {time.time()} (file-load-always)")
        else:
            if self.model is None:
                self.models[LLama3_model] = LlamaForCausalLM.from_pretrained(LLama3_model)
                self.models[LLama3_model].eval()   
                self.model = self.models[LLama3_model]
                print(f"LLama-3 Shuffle: Loaded LLama-3 from file at: {time.time()} (file-load-once)")

            if self.original_model is not None:
                del self.original_model
                torch.cuda.empty_cache()
                self.original_model = copy.deepcopy(self.model)
                print(f"LLama-3 Shuffle: LLama-3 model deepcopy saved at: {time.time()}")

        model = self.original_model
        return model


class LLama3ShuffleNode(LLama3ShuffleBase):
    def __init__(self, load_from_file=False):
        load_from_file=load_from_file
        self.models = {}
        self.model = None
        self.tokenizer = None
        self.original_model = None
        pass
        
    @classmethod
    def IS_CHANGED(c, **kwargs):
        sillytimestamp = time.time()
        return sillytimestamp

    @classmethod       
    def INPUT_TYPES(cls):      
        return {
            "required": {
                "text": ("STRING", {"default": "Describe a fantastical sci-fi robotic cat scene.", "multiline": True}),
                "max_response_length": ("INT", {"default": 128, "min": 50, "max": 1024}),
                "LLama3_model": (["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"], {"default": "meta-llama/Llama-3.2-1B-Instruct"}),
                "load_from_file": (["False", "True"], {"default": "True"}),
                "instruct_system_prompt": ("STRING", {"default": "You are a helpful assistant. You describe the image in great detail, focusing on lighting, mood, scene, details, subjects, and colors. Do not add anything else; only respond with the image description.", "multiline": True}),
                "shuffle_setting": (["None", "Attn", "Layer", "MLP", "LN_Identity"], {"default": "None"}),
                "shuffle_layer_range": ("STRING", {"default": "6,7,8"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "display": "number"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.01, "display": "number"}),
                "no_repeat_ngram_size": ("INT", {"default": 2, "min": 1, "max": 10}),
            },
        }
       
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "zer0int/LLama3-Shuffle"
    
   
    def generate(self, text, max_response_length, LLama3_model, load_from_file, instruct_system_prompt, shuffle_setting, shuffle_layer_range, temperature, top_p, no_repeat_ngram_size):
        
        self.load_from_file = load_from_file == "True"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        modelorg = self.get_original_model(LLama3_model) 
        model = modelorg.model

        num_return_sequences = 1
        max_length = max_response_length
        
        error_prompt = "empty"
        systemprompt = instruct_system_prompt
        
       
        num_layers = len(model.layers)       
        shuffle_layer_range_to_shuffle = [int(x.strip()) for x in shuffle_layer_range.split(",") if x.strip().isdigit()]
        
        if all(0 <= layer_idx < num_layers - 2 for layer_idx in shuffle_layer_range_to_shuffle):
        
            if shuffle_setting != "None":          
    
                for layer_idx in shuffle_layer_range_to_shuffle:
                    layer1 = model.layers[layer_idx]
                    layer2 = model.layers[layer_idx + 1]

                    if shuffle_setting == "MLP":
                        layer1.mlp.gate_proj.weight, layer2.mlp.down_proj.weight = layer2.mlp.gate_proj.weight, layer1.mlp.down_proj.weight
                        layer1.mlp.gate_proj.bias, layer2.mlp.down_proj.bias = layer2.mlp.gate_proj.bias, layer1.mlp.down_proj.bias
                        layer1.mlp.up_proj.weight, layer2.mlp.up_proj.weight = layer2.mlp.up_proj.weight, layer1.mlp.up_proj.weight
                        layer1.mlp.up_proj.bias, layer2.mlp.up_proj.bias = layer2.mlp.up_proj.bias, layer1.mlp.up_proj.bias

                    elif shuffle_setting == "Attn":
                        layer1.self_attn.q_proj.weight, layer2.self_attn.q_proj.weight = layer2.self_attn.q_proj.weight, layer1.self_attn.q_proj.weight
                        layer1.self_attn.q_proj.bias, layer2.self_attn.q_proj.bias = layer2.self_attn.q_proj.bias, layer1.self_attn.q_proj.bias
                        layer1.self_attn.k_proj.weight, layer2.self_attn.k_proj.weight = layer2.self_attn.k_proj.weight, layer1.self_attn.k_proj.weight
                        layer1.self_attn.k_proj.bias, layer2.self_attn.k_proj.bias = layer2.self_attn.k_proj.bias, layer1.self_attn.k_proj.bias
                        layer1.self_attn.v_proj.weight, layer2.self_attn.v_proj.weight = layer2.self_attn.v_proj.weight, layer1.self_attn.v_proj.weight
                        layer1.self_attn.v_proj.bias, layer2.self_attn.v_proj.bias = layer2.self_attn.v_proj.bias, layer1.self_attn.v_proj.bias
                        layer1.self_attn.o_proj.weight, layer2.self_attn.o_proj.weight = layer2.self_attn.o_proj.weight, layer1.self_attn.o_proj.weight
                        layer1.self_attn.o_proj.bias, layer2.self_attn.o_proj.bias = layer2.self_attn.o_proj.bias, layer1.self_attn.o_proj.bias

                    elif shuffle_setting == "Layer":
                        layer1, layer2 = layer2, layer1

                    elif shuffle_setting == "LN_Identity":
                        layer1.post_attention_layernorm = torch.nn.Identity()
                        layer2.post_attention_layernorm = torch.nn.Identity()
    
        else:
            error_prompt = f"ERROR: Specified Layers for LLama-3 out of range. Max for {LLama3_model}: {num_layers} -2 (don't shuffle the output!) = {num_layers - 2}"
            print(error_prompt)         
    
     
        modelorg.model = model
        model = modelorg    
        model = model.to(device)

        if LLama3_model == "meta-llama/Llama-3.2-1B":
            self.tokenizer = AutoTokenizer.from_pretrained(LLama3_model)
            self.tokenizer.eos_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1

            text = text.strip()            
            
            tokens = self.tokenizer(text, return_tensors="pt", padding=True)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]          
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_response_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                do_sample=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=-1.0,
            )
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            new_prompt = output_texts[0].strip()
            new_prompt = new_prompt.replace(text, "").strip()


        elif LLama3_model == "meta-llama/Llama-3.2-1B-Instruct":
            self.tokenizer = AutoTokenizer.from_pretrained(LLama3_model)
            self.tokenizer.eos_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1
            
            text = text.strip()

            pipe = pipeline(
                "text-generation", 
                model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16, 
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                do_sample=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                device=device,
            )

            messages = [
                {"role": "system", "content": f"{systemprompt}"},
                {"role": "user", "content": f"{text}"},
            ]
            
            outputs = pipe(
                messages,
                max_new_tokens=max_response_length,
            )

            generated_text_list = outputs[0]["generated_text"]
            assistant_content = next(
                (item["content"] for item in generated_text_list if item["role"] == "assistant"), 
                ""  # Default to an empty string if not found
            )

            clean_content = assistant_content.replace("\n", " ")
            clean_content = clean_content.replace("  ", " ")
            new_prompt = clean_content
   
        timestamp = time.time()  
    
        if error_prompt != "empty":
            return (error_prompt,)
        else:
            return (new_prompt,)

NODE_CLASS_MAPPINGS = {
    "LLama3ShuffleNode": LLama3ShuffleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLama3ShuffleNode": "LLama-3 Shuffle & Prompt",
}
TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'
pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'


llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"
mixtral_22b_name = "mistral-community/Mixtral-8x22B-v0.1"

from PIL import Image
import requests
import json

import torch
from pathlib import Path
import os
from os.path import join
import copy
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlavaConfig, LlavaForConditionalGeneration
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel



from llava.mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from llava.mixtral_modification.modeling_mixtral import MixtralAdapterModel, MixtralAdapterForCausalLM


#from llava.llava_modification.configuration_llava import LlavaConfig
#from llava.llava_modification.modeling_llava import LlavaForConditionalGeneration

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)


mixtral_config = MixtralAdapterConfig(
    vocab_size=32002,
    shared_adapter=True,
    shared_adapter_type='LoRA',
    shared_adapter_args={'r': 256, 'lora_alpha': 32, 'lora_dropout': 0.1},
    embedded_sparse_adapter=True,
    embedded_sparse_adapter_type='LoRA',
    embedded_sparse_adapter_args={'r': 256, 'lora_alpha': 32, 'lora_dropout': 0.1}
)


mixtral_config = MixtralAdapterConfig(
    vocab_size=32000,
    shared_sparse_adapter=True,
    shared_sparse_adapter_num_experts=4,
    shared_sparse_adapter_num_experts_per_tok=2,
    shared_sparse_adapter_type='LoRA',
    shared_sparse_adapter_args={'r': 256, 'lora_alpha': 32, 'lora_dropout': 0.1},
    embedded_sparse_adapter=True,
    embedded_sparse_adapter_type='LoRA',
    embedded_sparse_adapter_args={'r': 256, 'lora_alpha': 32, 'lora_dropout': 0.1}
)

'''
mixtral_config = MixtralConfig(
    shared_expert=True,
    shared_expert_type='Parallel Adapter',
    shared_expert_args={'hidden_dim': 16, 'dropout': 0.1},
    sparse_adapter=True,
    sparse_adapter_type='Parallel Adapter',
    sparse_adapter_args={'hidden_dim': 16, 'hidden_act': 'silu', 'dropout': 0.1}
)
'''

mixtral_model = MixtralAdapterForCausalLM.from_pretrained(
    '/home/vault/b207dd/b207dd11/cache/huggingface/transformers/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841',
    #mixtral_name, 
    config=mixtral_config,
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float32,
    cache_dir=TRANSFORMERS_CACHE_DIR,
    device_map="auto"
)


mixtral_model.resize_token_embeddings(32002)


save_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/mixtral-lora-shared+sparse_256/'
mixtral_model.save_pretrained(save_model_dir)


'''
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all"
inputs = model_processor.tokenizer(text, return_tensors="pt")
outputs = mixtral_model.generate(**inputs.to(mixtral_model.device), max_new_tokens=50)
model_processor.batch_decode(outputs)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'''





pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'

llava_config = LlavaConfig.from_pretrained(pretrained_model_dir)
llava_config.text_config = mixtral_config


model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    config=llava_config,
    device_map="auto",
    torch_dtype="auto"
)




model_processor = AutoProcessor.from_pretrained(pretrained_model_dir)


model.eval()

prompt = "<image>\nUSER: What are people doing in this image?\nASSISTANT:"
image_path = "/home/hpc/b207dd/b207dd11/llava/img/piano.jpg"
image = Image.open(image_path)
inputs = model_processor(text=prompt, images=image, return_tensors="pt")
for k,v in inputs.items():
    v = v.to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=50)
model_processor.batch_decode(generate_ids)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]







trainable_param_names = ['lora_A', 'lora_B', 'adapter_w1', 'adapter_w2']


def convert_trainable_parameters(model, trainable_param_names):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if any(substring in name for substring in trainable_param_names):
            print(name)
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            #print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


convert_trainable_parameters(model, trainable_param_names)
print_trainable_parameters(model)





save_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained_parallel-adapter_16/'
model.save_pretrained(save_model_dir)


model = LlavaForConditionalGeneration.from_pretrained(
    save_model_dir,
    device_map="auto",
    torch_dtype="auto"
)











print_trainable_parameters(model.multi_modal_projector)
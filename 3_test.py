import os 
import torch 
from random import randint
from datasets import load_dataset
from tqdm.auto import tqdm 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import argparse

parser = argparse.ArgumentParser(description="Run model inference on test dataset")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
args = parser.parse_args()

model_basename = args.model_name 

model_name = "./llama-3.1-korean-8b-hf-20-epoch/"+model_basename

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto"
)

test_dataset = load_dataset(
    "json",
    data_files=os.path.join("", "./train_dataset.json"),
    split="train",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


test_dataset = load_dataset("json", 
                            split="train",
                            data_files="test_dataset.json")

result = []
for index in tqdm(range(0, args.num_samples)):
    messages = test_dataset[index]["messages"][:2]

    terminators = [
        tokenizer.eos_token_id,
    ]

    # Test on sample 
    input_ids = tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    question = test_dataset[index]['messages'][1]['content']
    answer = test_dataset[index]['messages'][2]['content']
    generation = tokenizer.decode(response,skip_special_tokens=True)
    result.append([question, answer, generation])

with open(f"./test/model_generation_result_{model_basename}.txt", "w") as file:
    for line in result:
        file.write(str(line) + "\n")


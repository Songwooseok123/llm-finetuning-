import os
import torch
from accelerate import PartialState
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from peft import get_peft_config, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, load_from_disk, concatenate_datasets
from huggingface_hub import login
from datasets import Dataset, load_dataset

device_string = PartialState().process_index

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
quant_config8 = BitsAndBytesConfig(
    load_in_8bit=True)
login(
  token="토큰",
  add_to_git_credential=True
)
lora_config = LoraConfig(
    r=256,
    lora_alpha = 128,
    lora_dropout = 0.05,
    target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj"],
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    cache_dir = '/workspace/gemma-2-9b-it',
    #quantization_config=quant_config8,
    quantization_config=quant_config,
    device_map ={'':device_string},
    torch_dtype = torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="eager",
)
model.gradient_checkpointing_enable() ############################

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it",
    cache_dir = '/workspace/gemma-2-9b-it',
                                         use_fast=True)


model = prepare_model_for_kbit_training(model)

dataset = load_dataset("json", data_files="/workspace/train_dataset.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"].select(range(2000))
test_dataset = dataset["test"].select(range(20))

train_valid= train_dataset.train_test_split(test_size=0.1, seed=45)
train_dataset = train_valid["train"].select(range(1000))
valid_dataset = train_valid["test"].select(range(20))

sfttrainer_args = TrainingArguments(
        output_dir="results",
        num_train_epochs = 1,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=5,
        per_device_eval_batch_size = 5,
        optim="adamw_torch_fused",
        warmup_ratio=0.03,
        weight_decay = 0.01,
        max_grad_norm = 0.5,
        learning_rate = 2e-4,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        save_steps = 100,
)




trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    max_seq_length=512,
    args=sfttrainer_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)  

trainer.train()
trainer.model.save_pretrained("adapter_accelerate")

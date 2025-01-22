# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 ./lora_multi_gemma.py --config ./config.yaml
from dataclasses import dataclass, field
import logging
import random

import torch
import transformers
import trl
from huggingface_hub import login
import json 
import torch
from datasets import Dataset, load_dataset
from trl import (setup_chat_format, 
                 DataCollatorForCompletionOnlyLM, 
                 SFTTrainer)
from trl.commands.cli_utils import  TrlParser
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig 
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          TrainingArguments, 
                          BitsAndBytesConfig, 
                          pipeline, 

                          StoppingCriteria, set_seed)
login(
  
  add_to_git_credential=True
)


dataset = load_dataset("json", data_files="/workspace/train_dataset.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"].select(range(2000))
test_dataset = dataset["test"].select(range(20))

train_valid= train_dataset.train_test_split(test_size=0.1, seed=45)
train_dataset = train_valid["train"].select(range(500))
valid_dataset = train_valid["test"].select(range(20))

@dataclass 
class ScriptArguments:
    model_name: str = field(
    default=None, metadata={"help": "SFT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"}
    )

def training_function(script_args, training_args):    
    
    # Model 및 파라미터 설정하기 
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        cache_dir = '/workspace/gemma-2-9b-it',
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 얘는 flashattention2 쓰자 ㅇ
        #quantization_config = quant_config,
        #load_in_8bit=True, # 이거랑 밑에 둘다 안 됨 
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True),# -> 이건 qlora 실습 때 해결해보자
        use_cache=False if training_args.gradient_checkpointing else True,  
    )
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    #tokenizer.padding_side = 'right'
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 1):
            print(train_dataset[index])
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
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
    
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    max_seq_length=512,
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint) # checkpoint 부터 실험 재개할 수 있도록

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, TrainingArguments)) 
    script_args, training_args = parser.parse_args_and_config()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)
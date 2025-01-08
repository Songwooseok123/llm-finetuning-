import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,

)
from trl import DataCollatorForCompletionOnlyLM

from trl import setup_chat_format
from peft import LoraConfig

from trl import (
   SFTTrainer)


from sklearn.model_selection import train_test_split

# Load dataset from the hub

from huggingface_hub import login

login(
    token="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",
    add_to_git_credential=True
)

### 3.5.3. 데이터셋 준비 
dataset = load_dataset("beomi/KoAlpaca-v1.1a")
columns_to_remove = list(dataset["train"].features)

system_prompt = "당신은 다양한 분야의 전문가들이 제공한 지식과 정보를 바탕으로 만들어진 AI 어시스턴트입니다. 사용자들의 질문에 대해 정확하고 유용한 답변을 제공하는 것이 당신의 주요 목표입니다. 복잡한 주제에 대해서도 이해하기 쉽게 설명할 수 있으며, 필요한 경우 추가 정보나 관련 예시를 제공할 수 있습니다. 항상 객관적이고 중립적인 입장을 유지하면서, 최신 정보를 반영하여 답변해 주세요. 사용자의 질문이 불분명한 경우 추가 설명을 요청하고, 당신이 확실하지 않은 정보에 대해서는 솔직히 모른다고 말해주세요."
 
train_dataset = dataset.map(
    lambda sample: 
    { 'messages' : [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]}]
                   },
)

train_dataset = train_dataset.map(remove_columns=columns_to_remove,batched=False)
train_dataset = train_dataset["train"].train_test_split(test_size=0.1, seed=42)

train_dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
train_dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)


LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

### 3.5.4. Llama3 모델 파라미터 설정 
@dataclass 
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "데이터셋 파일 경로"
        },
    )
    model_name: str = field(
    default=None, metadata={"help": "SFT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"}
    )
    question_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 질문 키"}
    )
    answer_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 답변 키"}
    )


def training_function(script_args, training_args):    
    # 데이터셋 불러오기 
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train",
    )

    # Model 및 파라미터 설정하기 
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        attn_implementation="sdpa",  # 얘는 flashattention2 쓰자 ㅇ
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,  
    ) # gradient_checkpointing(메모리 효율, 시간 늘어남)랑 use_cache(시간 줌, 메모리 저장)는 동시에 못씀

    # 토크나이저 및 데이터셋 chat_template으로 변경하기      
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"}) ###
    model.config.pad_token_id = tokenizer.pad_token_id ###
    #tokenizer.pad_token = tokenizer.eos_token
    
    #tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.padding_side = 'right'
    
    
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # 데이터가 변화되었는지 확인하기 위해 2개만 출력하기 
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    
    
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    response_template = "assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # Train 설정 
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        #packing = True,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        },
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint) # checkpoint 부터 실험 재개할 수 있도록

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, TrainingArguments)) # TrlParser가 YAML 파일의 하이퍼파라미터들을 파싱하여,  ScriptArguments 및 TrainingArguments 객체에 값을 매핑한다.
    script_args, training_args = parser.parse_args_and_config()  # 예를 들어 YAML에서 model_name: "meta-llama/Llama-2-7b"라고 정의했다면, script_args.model_name에 자동으로 반영됩니다.
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)



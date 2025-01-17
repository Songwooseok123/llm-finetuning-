# 모델
- gemma-2-9b-it
- 8bit 양자화 -> 10.4GB
  
# Task 
내담자의 입력에 대한 상담을 생성하는 ai 
# 데이터
- !wget 으로 웹에서 파일 다운로드
- 상담사, 내담자 multiturn 대화데이터
- 전처리
```
Dataset({
  features: ['messages'],
  num_rows:8731
})

[{'role': 'user', 'content': '안녕하세요'},
{'role': 'assistant', 'content': '안녕하세요'}]
```

# 학습
- Method: Lora
```
peft_config = LoraConfig(
        lora_alpha=128, # 보통 alpha = 2r
        lora_dropout=0.05,
        r=256, # 표현력
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
```
- packing: SFTTrainer supports example packing, where multiple short examples are packed in the same input sequence to increase training efficiency
## single_gpu_train
## multi_gpu_train
- load_in_8bit = True 인자를 주니 fsdp 학습이 되지 않는다.
  - **"ValueError: Cannot flatten integer dtype tensors"**

# Inference
- pipeline 사용
- stopping_criteria 사용
# 평가 
- openAI로 평가하기
- 

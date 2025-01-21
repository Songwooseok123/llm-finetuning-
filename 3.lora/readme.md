# 모델
- gemma-2-9b-it
- 8bit 양자화 -> 10.4GB
  
# Task 
내담자의 입력에 대한 상담을 생성하는 ai 
# 데이터
- data_preprocess.ipynb
  - !wget 으로 웹에서 파일 다운로드
  - 상담사, 내담자 multiturn 대화데이터
 ```
    Dataset({
      features: ['messages'],
      num_rows:8731
    })
 
  - 전처리 결과
    dataset['train'][0]
 
  {'messages': [{'role': 'user', 'content': '내가 약간 중2병 같은 걸 증상을 보이고 있어요.'},
  {'role': 'assistant', 'content': '중2병 증상이라니, 어떤 증상이신 건가요?'},
  {'role': 'user',
   'content': '그러니까 공부하기 싫어하고, 공격적이고, 좀 무례하게 말하고 싶은 게 많아져서 그런 거예요.'},
  {'role': 'assistant',
   'content': '그런 증상이 있으니까 힘드시겠죠. 중2병 같은 것이라고 생각하시는 이유는 무엇인가요?'},
  {'role': 'user', 'content': '막 공부 안하고 이것저것 들먹이고 하고 싶은 게 너무 많아서 그런 거 같아요.'}]}
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

# Inference
- before_train.ipynb
- after_train.ipynb
- pipeline 사용
- stopping_criteria 사용

# 평가 
- openAI로 평가하기
- 

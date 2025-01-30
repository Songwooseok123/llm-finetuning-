## multi_gpu_train
- general 폴더: 양자화를 하지않고 학습
  - 다음 user의 발화까지 생성하는 문제가 있음.
  - 따라서 추론 시 stopping_criteria을 사용하여 user token이 등장하면 멈추는 방법을사용했음
- general_collator 폴더: DataCollatorForCompletionOnlyLM를 사용해서, 상담자의 말만 학습시키기.
  - 추론시 eos_token_id=terminators 인자를 주어, 상담자의 말이 끝나고 나서 학습된 padding토큰을 만나면 생성을 멈추게 했음  

- load_in_8bit = True 인자를 주니 fsdp 학습이 되지 않는다.
  - **"ValueError: Cannot flatten integer dtype tensors"**
  - load_in_8bit lora 폴더 따로 만들 예정.

# 필요한 라이브러리 설치하기
```
pip install -r requirements.txt
```

# 학습하기
- 0_full_fine_tuning_config.yaml
- 1_train_full_fine_tuning.py

위의 2개 파일을 가지고 밑의 코드를 터미널에 실행시키면 학습이 진행됨. 
```
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 ./1_train_full_fine_tuning.py --config ./0_full_fine_tuning_config.yaml
```
## 실행 스크립트 설명
- **ACCELERATE_USE_FSDP=1** : FSDP 사용하겠다
- **FSDP_CPU_RAM =1** : FSDP 사용시 CPU RAM을 효율적으로 사용해 모델을 로딩하겠다.
- **torchrun --nproc_per_node=2** : n개의 노드(컴퓨터, 즉 gpu)에서 프로세스를 (병렬적으로)실행한다.
  - torchrun : 스크립트를 GPU 여러 개에서 병렬로 실행하는 PyTorch의 기능
- **./1_train_full_fine_tuning.py** : 훈련 스크립트
- **./0_full_fine_tuning_config.yaml** : 처음 ./1_train_full_fine_tuning.py을 봤을 땐, yaml 파일의 내용(하이퍼파라미터)을 호출하는 부분이 없어보였음
  - ./1_train_full_fine_tuning.py 코드 속 **TrlParser의 역할**
    - **TrlParser**는 trl 라이브러리에서 제공하는 유틸리티로, 명령어에서 --config 뒤에 입력해서 전달된 YAML 파일을 읽고, 그 값을 TrainingArguments 및 사용자 정의 인자(ScriptArguments)에 자동으로 반영된다.

## 1_train_full_fine_tuning.py
- 파이썬 파일을 첫줄부터 실행시킨다.
- if __name__ == "__main__": 직접 파이썬 스크립트를 터미널에서 실행시켰기 때문에 조건문 이하 코드를 각 gpu에서 실행시킨다. 
  - yaml 파일의 하이퍼 파라미터를 TrlParser를 통해 전달해서 script_args, training_args를 만들고, training_function(script_args, training_args)을 실행시킨다.
```
############ import 블록 ############
import logging
# 중략...

############ dataset 블록 ############
dataset = # 중략
#...

############ dataclass 블록 ############
@dataclass
class ScriptArguments:
# 중략...

############ training_function블록 ############
def training_function(script_args, training_args):
  # 중략
  trainer.train(resume_from_checkpoint=checkpoint)

############ "__main__"블록 ############
if __name__ == "__main__":
# 얘는 내가 파이썬 스크립트를 직접 실행시킬 때만 만족함("__name__"이 "__main__"이 됨). 예를 들어 py 파일을 import 해와서 쓸 때는 실행되지 않는 부분임 

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)
```


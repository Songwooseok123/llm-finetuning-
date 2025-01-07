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
- **./1_train_full_fine_tuning.py** : 스크립트
- **./0_full_fine_tuning_config.yaml** : 처음 ./1_train_full_fine_tuning.py을 봤을 땐, yaml 파일의 내용(하이퍼파라미터)을 호출하는 부분이 없어보였음
  - **TrlParser의 역할**
    - **TrlParser**는 trl 라이브러리에서 제공하는 유틸리티로, 명령어에서 --config로 전달된 YAML 파일을 읽고, 그 값을 TrainingArguments 및 사용자 정의 인자(ScriptArguments)에 매핑합니다.
    - --config 뒤에 0_full_fine_tuning_config.yaml을 넣으면, 해당 파일이 자동으로 파싱되어 script_args와 training_args에 반영됩니다.
    - TrlParser가 --config 뒤에 있는 0_full_fine_tuning_config.yaml을 감지하고, 이 파일을 불러옵니다.
    - YAML 파일의 내용은 TrainingArguments 및 ScriptArguments에 자동으로 반영됩니다.

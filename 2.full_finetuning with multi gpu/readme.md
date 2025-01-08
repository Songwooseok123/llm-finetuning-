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
- if __name__ == "__main__": 직접 파이썬 스크립트를 터미널에서 실행시켰기 때문에 조건문 이하 코드를 **각 gpu**에서 실행시킨다. 
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

## dataclass 블록에 class ScriptArguments -> 스크립트 인자 정의 
- **TrlParser**가 --config로 전달된 YAML 파일(0_full_fine_tuning_config.yaml)을 파싱하여,
- ScriptArguments 및 TrainingArguments 객체에 값을 매핑합니다.
- 예를 들어, YAML에서 model_name: "meta-llama/Llama-2-7b"라고 정의했다면, script_args.model_name에 자동으로 반영됩니다.
### @dataclass 데코레이터 
- 파이썬의 dataclass를 사용해, ScriptArguments라는 클래스를 정의하고, 이 클래스에서 스크립트 실행 시 필요한 매개변수(필드)를 관리한다.
- dataclass 데코레이터를 안 쓰면, init, repr, eq 등을 직접 작성해야 한다.
- datclass가 자동으로 위의 것들을 생성해준다.
- field 함수: 각 필드를 세부적으로 설정하는데 사용. 
  - metadata: 필드에 대한 부가 정보

**"yaml 파일에서 파라미터 필드 정의하고, py 파일에서 class ScriptArgument에 필드 만들고, TrlParser로 script_args 만들어서 가져다가 쓰면됨."**

## training_function블록 
### gradient_checkpointing(메모리 효율, 시간 늘어남)랑 use_cache(시간 줌, 메모리 저장)는 동시에 못씀

- task에 따라서 collator 신경쓰기  
  - 아근데, collator 쓰니까 packing을 못하네
  - packing =True를 지우니까, step 수가 좀 늘어남.
    - SFTTrainer supports example packing, where multiple short examples are packed in the same input sequence to increase training efficiency
   
  - -> collator 추가하니까, 반복되는 문제 발생
  - collator 빼고 다시 학습 돌려보니까 괜찮음
  - collator 추가하고, pad_token id 바꿔서 다시 돌려보기

 반복, 종결x  문제 발생하는지 실험 
| 기본(without collator)  |  with collator | with collator & reverse padding token |
|--------------|--------------------------|--------------|
| 양호     | 문제발생                 | 어느정도 문제 해결|
- 위의 표 완성하려면, test.py로 50개씩 돌려서 확인해야겠다. 

## **자 그럼, 앞으로 학습을 할 때, collator가 필요하다면 pad_token을 바꿀지 말지 정해야 겠다. **


# wand 
- train/grad_norm 학습이 안정적으로 진행되고 있는지
- ![image](https://github.com/user-attachments/assets/07a5d71d-f4e9-4926-8664-b25be3009c03)


# 파이썬 스크립트로 한 번에 1000개 추론하기. 
- test 폴더 만들고, 모델 별로 실행
```
python 3_test.py --model_name checkpoint-148 --num_samples 50
python 3_test.py --model_name checkpoint-296 --num_samples 50
python 3_test.py --model_name collator_checkpoint-148 --num_samples 50
python 3_test.py --model_name collator_checkpoint-297 --num_samples 50
python 3_test.py --model_name without_col_checkpoint-127 --num_samples 50
python 3_test.py --model_name without_col_checkpoint-254 --num_samples 50
```
# openai_test.py
- requirements_for_gptEval.txt 설치 하고
- python 4_openai_test.py 하면
- sample 별로 평가가 나옴. (qa_evaluation_results 폴더에 생성)
- 내가 prompt랑 class Evaluation(BaseModel): 수정해서 평가 항목 추가할 수 있음. 6은 내가 추가 한거임. 

# 샘플별로 평가한 것을 종합적으로 평가 -> 모델평가 
5_score_notebook.ipynb 실행 : qa_evaluation_results 폴더에 있는 값 가져와서 계산 

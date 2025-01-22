- 모델성능 openAI로 평가하기.ipynb
  - seed_text 전처리 
    - def simulate_conversation(pipeline, num_turns =10):
      -  **conversation의 마지막 user 말만 pipeline의 input으로 들어가는거 아닌가 여기서?**
    - def read_conversation(file_path: str) -> List[str]:
      -   ![image](https://github.com/user-attachments/assets/01274e6e-b1a7-41eb-a21f-e19ce3444ba3)
  - conversation 넣고, json으로 응답받기
    - json 형식 벗어 났을 떄, json으로 바꾸는거랑 ,예외처리하는거 다 포함되어 있음.  

- 나는 모델을 학습시킬 때, colltor 없이 학습을 시켰다, 그래서 model의 발화와 user의 발화를 모두 생성한다.
- 따라서 평가를 할 때 stopping_criteria(user 나오면 생성 멈추게)와 밑의 함수를 적용해서 multiturn 대화를 뽑아냈다.(결과물의 끝에서 user제거하기)
  ```
  def remove_suffix_if_exists(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[:-len(suffix)]  # suffix 길이만큼 잘라냄
    return text
  ```
## colltor를 써서 "내담자+모델+내담자..." 입력이 들어왔을 때, 상담자(model)의 발화만 출력할 수 있도록 학습을 시켜봐야겠다. 

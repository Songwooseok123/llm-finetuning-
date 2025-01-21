- 모델성능 openAI로 평가하기.ipynb
  - seed_text 전처리 
    - def simulate_conversation(pipeline, num_turns =10):
      -  **conversation의 마지막 user 말만 pipeline의 input으로 들어가는거 아닌가 여기서?**
    - def read_conversation(file_path: str) -> List[str]:
      -   ![image](https://github.com/user-attachments/assets/01274e6e-b1a7-41eb-a21f-e19ce3444ba3)
  - conversation 넣고, json으로 응답받기
    - json 형식 벗어 났을 떄, json으로 바꾸는거랑 ,예외처리하는거 다 포함되어 있음.  



# 실헙결과
## after_train_no_quantization.ipynb 

4bit 양자화 모델로 lora adapter를 얻은 후 추론결과

- 데이터 1000개로 학습
  - 모델 로드 시 양자화를 안 하고 로드한 후 adapter를 붙혔을 경우 결과가 좋다
## after_train_quantization.ipynb 
  - 하지만 오히려 4bit로 로드한 후 adapter를 붙이니 학습이 거의 안 된 것 같다.
  - 따라서 학습 데이터를 늘려 보았다.
  - 그러자 

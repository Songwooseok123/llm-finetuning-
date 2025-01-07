- 0_full_fine_tuning_config.yaml
- 1_train_full_fine_tuning.py

위의 2개 파일을 가지고 밑의 코드를 터미널에 실행시키면 학습이 진행됨. 

```
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 ./1_train_full_fine_tuning.py --config ./0_full_fine_tuning_config.yaml
```

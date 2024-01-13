# HRL-HF_risk_factor_set
## 1 How to config the environments:
- on Ubuntu 20.04.5
- python 3.8
- numpy 1.24.2
- pandas 1.1.5
- pytorch 1.7.1
- tensorboard 2.14.0
- scipy 1.10.1
## 2 How to train model:
### 2.1 HPPO-TO: 
```python python run_main_hppo_to.py ```
### 2.2 HPPO/DAC:
```python python run_main_hrl.py ```
### 2.3 PPO:     
```python python run_main_ppo.py ```
#### For the above HRL/RL model use .npz file which contains ```python open,high,low,close,vwap,target``` as input. The dataset is of two types i.e. Environment name can be [sh300, sp500].

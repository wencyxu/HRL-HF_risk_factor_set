# HRL-HF_risk_factor_set
## 1 How to config the environments:
- on Ubuntu 20.04.5
- python 3.8
- numpy 1.24.2
- pandas 1.1.5
- pytorch 1.7.1
- tensorboard 2.14.0
- scipy 1.10.1
## 2  Prepare you training data
* For the above HRL/RL model use .npz file which contains ```open,high,low,close,vwap,target``` as input. The dataset is of two types i.e. ```arg.add_arg("env_name", "sh300", "Environment name, can be [sh300, sp500]")```.
* There are three ways to calculate rewards (a measure of the three metrics that generate the mathematical expression and target of a high-frequency risk factor) in ```class Risk_Factor_Env: IC*, RankIC*, and IR*```.
## 3 How to train model:
### 3.1 HPPO-TO: 
```python run_main_hppo_to.py ```
### 3.2 HPPO/DAC:
```python run_main_hrl.py ```
### 3.3 PPO:     
```python run_main_ppo.py ```


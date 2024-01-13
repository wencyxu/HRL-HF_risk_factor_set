import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from heapq import nlargest
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def load_weight_encoding():
    np.random.seed(0)
    weight_encoding = []
    for _ in range(6):
        weights = np.random.rand(5)
        weights += 0.1
        weights /= np.sum(weights)
        weight_encoding.append(weights.tolist())
    return weight_encoding


operator_encoding = list(itertools.permutations(['+', '-', '*', '/', '**'], 4))

def load_data(file_path):
    npz = np.load(file_path)
    data = npz['data']
    columns = npz['columns']
    df = pd.DataFrame(data, columns=columns)

    x = df[['open', 'low', 'high', 'close', 'vwap']]
    ret = df['future_RV']
    y = np.where(np.isnan(ret), 0, ret)

    return np.asarray(x), np.asarray(y)

class Risk_Factor_Env(object):
    def __init__(self, file_name):
        data = load_data('./data/{}.npz'.format(file_name))
        self.input_feature = data[0]
        self.future_RV = data[1]
        self.seq_len, self.feature_num = self.input_feature.shape
        self.feature_names = ['open', 'low', 'high', 'close', 'vwap']
        self.weight_encoding = load_weight_encoding()  # 加载 weight_encoding
        self.option_num = len(self.weight_encoding) 
        self.action_num = len(operator_encoding) 
        self.reset()
        self.future_RVs = []
        self.best_expressions = []
        self.values = []  
        self.ic_list = []
        self.counter = 0 
      
    def reset(self):
        self.t = np.random.randint(self.seq_len)
        self.end_time = self.t + 150
        self.high_state = self.input_feature[self.t]
        self.low_state = self.input_feature[self.t]
        self.done = False
        return self.input_feature[self.t]

    def step(self, high_option, low_action):
        self.high_option = high_option
        self.low_action = low_action

        weight_num = self.weight_encoding[high_option]
        operator_num = operator_encoding[low_action]
        feature_num = np.asarray(self.low_state) * np.asarray(weight_num)
        numeric_expression = self.generate_numeric_expression(feature_num, operator_num)
        label_expression = self.generate_label_expression(weight_num, operator_num, high_option)
        #label_expression = self.generate_label_expression(weight_num, operator_num)
        expression_value = self.evaluate_expression(numeric_expression)
        self.future_RVs.append(self.future_RV[self.t])
        self.values.append(expression_value)
        reward = self.calculate_IC(self.values, self.future_RVs)
        self.best_expressions.append(numeric_expression)
        self.t += 1
        if self.t >= min(self.end_time, self.seq_len):
            self.done = True
        else:
            self.high_state = self.input_feature[self.t]
            self.low_state = self.input_feature[self.t]

        if self.done:
            valid_expressions = [(value, expression) for value, expression in zip(self.values, self.best_expressions) if value != float('inf')]
            top_10_expressions = nlargest(10, valid_expressions)
            for value, expression in top_10_expressions:
                print(f'Numeric Expression: {expression}, Label Expression: {label_expression}, Value: {value}')
                # pass
        return self.high_state, reward, self.done
    
    def generate_numeric_expression(self, feature_num, operator_num):
        expression = str(feature_num[0])  
        for i in range(1, len(feature_num)):  
            expression += operator_num[i-1] + str(feature_num[i]) 
        return expression 

    def generate_label_expression(self, weight_num, operator_num, high_option):
        expression = '(w1*' + self.feature_names[0] + ')'
        for i in range(1, len(weight_num)):
            expression += operator_num[i-1] + '(w' + str(i+1) + '*' + self.feature_names[i] + ')'
        expression += f' (weight_encoding index: {high_option})'  
        return expression

    def evaluate_expression(self, expression):
        try:
            value = eval(expression) 
        except:
            value = float('inf') 
        return value
    


    '''case1 reward:IC(expression_values, future_RVs)'''
    def calculate_IC(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        ic, _ = stats.pearsonr(values, future_RVs)
        print(f"rank_ic type: {type(rank_ic)}, rank_ic value: {rank_ic}")
        if self.counter < 300:  
            self.counter += 1
        return ic
    
    
    '''case2 reward:RankIC(expression_values, future_RVs)'''
    '''
    def calculate_RankIC(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        rank_ic, _ = stats.spearmanr(values, future_RVs)
        if self.counter < 300:
            print(f"rank_ic type: {type(rank_ic)}, rank_ic value: {rank_ic}")  
            self.counter += 1
        return rank_ic
    '''
    
    '''case3 reward:IR(expression_values, future_RVs)'''
    '''
    def calculate_IR(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        ic, _ = stats.pearsonr(values, future_RVs)
        self.ic_list.append(ic)
        if len(self.ic_list) > 1:
            ir = np.mean(self.ic_list) / np.std(self.ic_list)
        else:
            ir = 0.0
        if self.counter < 300:
            print(f"ir type: {type(ir)}, ir value: {ir}")  
            self.counter += 1
        return ir
    '''
    
    def state_option_action_size(self):
        return self.feature_num, self.option_num, self.action_num
    
    

import numpy as np
import pandas as pd
import itertools
from scipy import stats
import os
import random
from numpy import isfinite
from heapq import nlargest
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def load_weight_encoding():
    weight_encoding = [
        [0.3, 0.09, 0.3, 0.1, 0.2, 0.01],
        [0.1, 0.3, 0.01, 0.01, 0.18, 0.4],
        [0.2, 0.01, 0.01, 0.01, 0.47, 0.3],
        [0.5, 0.46, 0.01, 0.01, 0.01, 0.01],
        [0.1, 0.1, 0.5, 0.2, 0.99, 0.01]
    ]
    return weight_encoding


'''
operator_encoding = list(itertools.permutations(['+', '-', '*', '/', '**',
                                                  '*sin', '*cos', '*tan', '*sqrt', '*arctan', 
                                                  ], 5))
'''
operator_encoding = list(itertools.product(['+', '-', '*', '/', '**'], repeat=5)) + \
                    list(itertools.product(['*sin', '*cos', '*tan', '*sqrt', '*arctan'], repeat=5))



def load_data(file_path):
    df = pd.read_csv(file_path)  
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    x = df[['open', 'low', 'high', 'close', 'volume', 'vwap']]
    ret = df['target']
    y = np.where(pd.isnull(ret), 0, ret)
    return np.asarray(x), np.asarray(y)

class Risk_Factor_Env(object):
    def __init__(self, file_name):
        
        data = load_data("/root/data/1min/HS300.csv")
        
        self.input_feature = data[0]
        self.target = data[1]
        self.seq_len, self.feature_num = self.input_feature.shape
        self.feature_names = ['open', 'low', 'high', 'close', 'volume','vwap']
        self.weight_encoding = load_weight_encoding()  
        self.action_num = len(operator_encoding) 
        self.option_num = len(self.weight_encoding)
        self.reset()
        self.targets = []
        self.best_expressions = []
        self.values = []  
        self.ic_list = []
        self.counter = 0 
     
    def reset(self):
        self.t = np.random.randint(self.seq_len)
        self.end_time = self.t + 100
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
        label_expression = self.generate_label_expression(weight_num, operator_num)  # Added line
        expression_value = self.evaluate_expression(numeric_expression)
        self.targets.append(self.target[self.t])
        self.values.append(expression_value)
        reward = self.calculate_IC(self.values, self.targets)
        self.best_expressions.append(numeric_expression)
        self.t += 1
        if self.t >= min(self.end_time, self.seq_len):
            self.done = True
        else:
            self.high_state = self.input_feature[self.t]
            self.low_state = self.input_feature[self.t]
        if self.done:            
            valid_expressions = [(value, expression) for value, expression in zip(self.values, self.best_expressions) if value != float('inf')]        
            top_3_expressions = nlargest(3, valid_expressions)
            for value, expression in top_3_expressions:
                print(f'Label Expression: {label_expression}')
                # print(f'Numeric Expression: {expression}, Label Expression: {label_expression}, Value: {value}')  # Modified line
        return self.high_state, reward, self.done

    def generate_numeric_expression(self, feature_num, operator_num):
        expression = str(feature_num[0])  
        for i in range(1, len(feature_num)):  
            if operator_num[i-1] in ['*sin', '*cos', '*tan', '*sqrt', '*arctan']:  
                expression += operator_num[i-1] + '(' + str(feature_num[i]) + ')'  
            else:
                expression += operator_num[i-1] + str(feature_num[i])  
        return expression  


    def generate_label_expression(self, weight_num, operator_num):
        expression = '(' + str(weight_num[0]) + '*' + self.feature_names[0] + ')'
        for i in range(1, len(weight_num)):
            if operator_num[i-1] in ['*sin', '*cos', '*tan', '*sqrt', '*arctan']:
                expression += operator_num[i-1] + '(' + str(weight_num[i]) + '*' + self.feature_names[i] + ')'
            else:
                expression += operator_num[i-1] + '(' + str(weight_num[i]) + '*' + self.feature_names[i] + ')'
        return expression

    def evaluate_expression(self, expression):
        try:
            value = eval(expression)  
        except:
            value = float('inf')  
        return value
    
    '''case1 reward: IC(expression_values, targets)'''

    def calculate_IC(self, values, targets): 
        if len(values) < 2 or len(targets) < 2:
            #print("Warning: the length of values or targets is less than 2.")
            return 0
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            #print("Warning: values or targets contain infinite or NaN values.")         
            values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            targets = np.nan_to_num(targets, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        ic, _ = stats.pearsonr(values, targets)
        return ic
       
    
    '''case2 reward:RankIC(expression_values, targets)'''
    '''
    def calculate_RankIC(self, values, targets):
        if len(values) < 2 or len(targets) < 2:
            return 0.0
        values = np.array(values)
        targets = np.array(targets)
        
        rank_ic, _ = stats.spearmanr(values, targets)
        
        if self.counter < 300:
            print(f"rank_ic type: {type(rank_ic)}, rank_ic value: {rank_ic}")  
            self.counter += 1
        return rank_ic
    '''
    
    '''case3 reward:IR(expression_values, targets)'''
    '''
    def calculate_IR(self, values, targets):
        if len(values) < 2 or len(targets) < 2:
            return 0.0
        values = np.array(values)
        targets = np.array(targets)
        
        ic, _ = stats.pearsonr(values, targets)
        
        self.ic_list.append(ic)
       
        if len(self.ic_list) > 1:
            ir = np.mean(self.ic_list) / np.std(self.ic_list)
        else:
            ir = 0.0
        
        if self.counter < 400:
            print(f"ir type: {type(ir)}, ir value: {ir}")  
            self.counter += 1
        return ir
    
    '''
    
    def state_option_action_size(self):
        
        return self.feature_num, self.option_num, self.action_num
    




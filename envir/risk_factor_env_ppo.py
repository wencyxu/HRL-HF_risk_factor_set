import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from heapq import nlargest
from scipy.stats import pearsonr

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
        
        self.action_num = len(operator_encoding) 
        self.reset()
        
        self.future_RVs = []
        self.best_expressions = []
        self.values = [] 
        self.ic_list = []
        self.counter = 0
        
    def reset(self):
        self.t = np.random.randint(self.seq_len)
        self.end_time = self.t + 200
        self.state = self.input_feature[self.t]
        self.done = False
        return self.input_feature[self.t]

    def step(self, action):
        self.action = action
        operator_num = operator_encoding[action]
        feature_num = np.asarray(self.state)
        numeric_expression = self.generate_numeric_expression(feature_num, operator_num)
        expression_value = self.evaluate_expression(numeric_expression)
        self.future_RVs.append(self.future_RV[self.t])
        self.values.append(expression_value) # ?
        reward = self.calculate_IR(self.values, self.future_RVs)
        self.best_expressions.append(numeric_expression)
        self.t += 1
        if self.t >= min(self.end_time, self.seq_len):
            self.done = True
        else:
            self.state = self.input_feature[self.t]
        if self.done:
            valid_expressions = [(value, expression) for value, expression in zip(self.values, self.best_expressions) if value != float('inf')] # ?
            top_3_expressions = nlargest(3, valid_expressions)
            for value, expression in top_3_expressions:
                print(f'Numeric Expression: {expression}, Value: {value}')
                #pass
        return self.state, reward, self.done
    
    def generate_numeric_expression(self, feature_num, operator_num):
        expression = str(feature_num[0])  
        for i in range(1, len(feature_num)):  
            expression += operator_num[i-1] + str(feature_num[i])  
        return expression  

    def evaluate_expression(self, expression):
        try:
            value = eval(expression)  
        except:
            value = float('inf')  
        return value

    '''case1 reward:IC*(expression_values, future_RVs)'''
    '''
    def calculate_IC(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        ic, _ = stats.pearsonr(values, future_RVs)
        if self.counter < 300:
            print(f"ic type: {type(ic)}, ic value: {ic}")  
            self.counter += 1
        return ic
    '''
    
    '''case2 reward:RankIC*(expression_values, future_RVs)'''
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
    

    def state_action_size(self):
        return self.feature_num, self.action_num


import numpy as np
import pandas as pd
import itertools
from scipy import stats
import os
from heapq import nlargest
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def load_weight_encoding():
    # 设置随机种子
    np.random.seed(0)
    # 初始化权重组合列表
    weight_encoding = []
    # 生成6个权重组合
    for _ in range(6):
        # 生成5个随机数
        weights = np.random.rand(5)
        # 确保随机数不为0
        weights += 0.1
        # 归一化权重
        weights /= np.sum(weights)
        # 将权重组合添加到列表中
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
        data = load_data('/home/zhanghy/xuwenyan/HPPO_init_test2/data/{}.npz'.format(file_name))
        # data = load_data('E:/2023Oct/HTF_risk_factor/HPPO_init_test2/data/{}.npz'.format(file_name))
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
        self.values = []  # 新增一个列表来收集所有的value
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
            # 创建一个新的列表，其中只包含非inf的expression_value
            valid_expressions = [(value, expression) for value, expression in zip(self.values, self.best_expressions) if value != float('inf')]
            # 从这个列表中获取最大的三个值
            top_10_expressions = nlargest(10, valid_expressions)
            for value, expression in top_10_expressions:
                print(f'Numeric Expression: {expression}, Label Expression: {label_expression}, Value: {value}')
                # pass
        return self.high_state, reward, self.done
    
    def generate_numeric_expression(self, feature_num, operator_num):
        expression = str(feature_num[0])  # 初始化表达式
        for i in range(1, len(feature_num)):  # 遍历剩余的特征
            expression += operator_num[i-1] + str(feature_num[i])  # 添加运算符和特征
        return expression  # 返回表达式
    '''
    def generate_label_expression(self, weight_num, operator_num):
        expression = '(w1*' + self.feature_names[0] + ')'
        for i in range(1, len(weight_num)):
            expression += operator_num[i-1] + '(w' + str(i+1) + '*' + self.feature_names[i] + ')'
        return expression
    '''
    def generate_label_expression(self, weight_num, operator_num, high_option):
        expression = '(w1*' + self.feature_names[0] + ')'
        for i in range(1, len(weight_num)):
            expression += operator_num[i-1] + '(w' + str(i+1) + '*' + self.feature_names[i] + ')'
        expression += f' (weight_encoding index: {high_option})'  # 添加weight_encoding的索引
        return expression

    def evaluate_expression(self, expression):
        try:
            value = eval(expression)  # 计算表达式的值
        except:
            value = float('inf')  # 如果表达式无法计算，返回无穷大
        return value
    

    '''case1 reward:IC(expression_values, future_RVs)'''
    
    def calculate_IC(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        # 计算皮尔逊相关系数
        ic, _ = stats.pearsonr(values, future_RVs)
        # 添加计数器
        if self.counter < 20:
            print(f"ic type: {type(ic)}, ic value: {ic}")  # 添加调试代码
            self.counter += 1
        return ic
    
    
    '''case2 reward:RankIC(expression_values, future_RVs)'''
    '''
    def calculate_RankIC(self, values, future_RVs):
        if len(values) < 2 or len(future_RVs) < 2:
            return 0.0
        values = np.array(values)
        future_RVs = np.array(future_RVs)
        # 计算斯皮尔曼等级相关系数
        rank_ic, _ = stats.spearmanr(values, future_RVs)
        # 添加计数器
        if self.counter < 100:
            print(f"rank_ic type: {type(rank_ic)}, rank_ic value: {rank_ic}")  # 添加调试代码
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
        # 计算皮尔逊相关系数
        ic, _ = stats.pearsonr(values, future_RVs)
        # 添加到IC列表
        self.ic_list.append(ic)
        # 计算IR
        if len(self.ic_list) > 1:
            ir = np.mean(self.ic_list) / np.std(self.ic_list)
        else:
            ir = 0.0
        # 添加计数器
        if self.counter < 16000:
            print(f"ir type: {type(ir)}, ir value: {ir}")  # 添加调试代码
            self.counter += 1
        return ir
    '''
    
    def state_option_action_size(self):
        # 返回高层策略的状态和选项的维度
        return self.feature_num, self.option_num, self.action_num
    
    
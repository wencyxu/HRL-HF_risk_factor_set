#!/usr/bin/env python3
# 直接print result
import os
import numpy as np
import pandas as pd 
import scipy.stats as stats
from scipy.stats import spearmanr
from model.GP.genetic import get_correlation_metrics
from model.GP.genetic import SymbolicTransformer

import warnings
warnings.filterwarnings("ignore",category = RuntimeWarning)


def preprocess():
    # file_path = "/root/HPPO_venv/HPPOvenv/HPPO_init_test2/data/sp500_80npz/80.npz"
    file_path = "/root/HPPO_venv/HPPOvenv/HPPO_init_test2/data/sh300_80npz/80.npz"
    npz = np.load(file_path, allow_pickle=True)
    data = npz['data']
    columns = npz['columns']
    df = pd.DataFrame(data, columns=columns)
    
    x = df[['open','low','high','close','vwap']]
    ret = df['future_RV']
    y = np.where(np.isnan(ret), 0, ret)

    '''
    # split sh300 dataset
    x_train = x.iloc[1:48901]    
    y_train = y[1:48901]          
    x_eval = x.iloc[48901:62101]  
    y_eval = y[48901:62101]       
    x_test = x.iloc[62101:]       
    y_test = y[62101:]            
    
    # split sp500 dataset
    
    x_train = x.iloc[1:49934]   
    y_train = y[1:49934]          
    x_eval = x.iloc[40034:62791]  
    y_eval = y[40034:62791]       
    x_test = x.iloc[62791:]       
    y_test = y[62791:]            
    '''
    # split sh300 dataset
    x_train = x.iloc[1:700]    
    y_train = y[1:700]          
    x_eval = x.iloc[700:800]  
    y_eval = y[700:800]       
    x_test = x.iloc[800:]       
    y_test = y[800:]  
    
    return x_train, y_train, x_eval, y_eval, x_test, y_test

def main():
    x_train, y_train, x_eval, y_eval, x_test, y_test = preprocess()
    func = ['add','sub','mul','div','sqrt']
    ST_gplearn = SymbolicTransformer(
        generations=30,
        n_components=1,
        hall_of_fame=80,    
        p_crossover=0.9,
        metric='pearson',
        const_range=None,
        init_depth=(2,4),
        tournament_size=10,
        function_set=func,
        p_point_replace=0.4,
        population_size=200,
        p_hoist_mutation=0.01,
        p_point_mutation=0.01,
        p_subtree_mutation=0.01,
        parsimony_coefficient=0,
        feature_names=['open','low','high','close','vwap'],
        random_state=1)
    
    ST_gplearn.fit(x_train, y_train)
    best_programs_dict = {}

    best_programs = ST_gplearn._best_programs

    result_dict = {"Expression": [],  "IC": [], "RankIC": []} # "Values": [],

    for bp in best_programs:
        factor_name = 'alpha_' + str(best_programs.index(bp) + 1)
        expression_values = bp.execute(x_train.values)
        #ic = stats.pearsonr(expression_values, y_train)[0]
        #rank_ic = stats.spearmanr(expression_values, y_train)[0]
        # 获取度量函数
        metrics = get_correlation_metrics()
        # 计算 ic 和 rank_ic
        ic = metrics["pearson"](expression_values, y_train)
        rank_ic = metrics["spearman"](expression_values, y_train)
        best_programs_dict[factor_name] = {'fitness':bp.fitness_, 'expression':str(bp), 'depth':bp.depth_, 'length': bp.length_, 'values': expression_values, 'IC': ic, 'RankIC': rank_ic}

        result_dict["Expression"].append(str(bp))
        # result_dict["Values"].append(expression_values)
        result_dict["IC"].append(ic)
        result_dict["RankIC"].append(rank_ic)
 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    result_df = pd.DataFrame(result_dict)
    # print(result_df)
    print(result_df.to_string())

if __name__ == "__main__":
    main()



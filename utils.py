# -*- coding:utf-8 -*-
'''
作者：ych
日期：2023年08月15日
'''
import pandas as pd
import matplotlib.pyplot as plt

path_train=r'./results_2_8/trainset_prediction_'
path_test=r'./results_2_8/testset_prediction_'
path_train2=r'./results_2_8/t_vs_p_train/tp_train_'
path_test2=r'./results_2_8/t_vs_p_test/tp_test_'

list_model=["LR","KNN","SVR","Ridge","Lasso","MLP","DT","ET","RF","Ada","GB","bag"]

def figure1(d1,d2,name,path=None):
    plt.figure(facecolor='gray', figsize=[16, 8])
    ##plot true
    plt.plot(d1, '-', label=name+" true", linewidth=1,color='b')
    plt.plot(d1, 'o',color='b')
    ##plot predicted
    plt.plot(d2, '-', label=name + " predicted", linewidth=1,color='r')
    plt.plot(d2, 'o',color='r')
    plt.grid()
    plt.title(name+" true vs predicted", fontsize=20)
    plt.legend(fontsize=16)
    if path:
        ##保存figure
        plt.savefig(path)
    else:
        ##展示figure
        plt.show()


if __name__ == '__main__':
    for m in list_model:
        path1=path_train+m+".csv"
        path2=path_train2+m+".png"
        data_train=pd.read_csv(path1)
        d1 = data_train["Truth"].values
        d2 = data_train["Prediction"].values
        figure1(d1, d2, m+" for trainset",path2)

        path1=path_test+m+".csv"
        path2=path_test2+m+".png"
        data_test=pd.read_csv(path1)
        d3=data_test["Truth"].values
        d4=data_test["Prediction"].values
        figure1(d3,d4,m+" for testset",path2)
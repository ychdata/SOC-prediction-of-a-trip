# -*- coding:utf-8 -*-
'''
作者：ych
日期：2023年08月13日
'''
###用基本的回归模型处理，预测2-3月的耗电与温度,load的数据集
##Y:最后一列soc_ave
##X:[1,2,3,4,5]:[load_ave,load_max,load_min,T_ave,T_max]
import pandas as pd
import numpy as np
import os
import argparse   ##记录参数
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA   ##无监督的特征降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  ##有监督的特征降维
from models_regression import model_pre  ##调用模型
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# path=r'../dataprocess(包含了data)/describe_analysis/trip_all2_2_8.csv'
path = r''   ##数据文件的路径
list_model=["LR","KNN","SVR","Ridge","Lasso","MLP","DT","ET","RF","Ada","GB","bag"]
dict_model={"LR":"线性回归","KNN":"KNN","SVR":"SVR","Ridge":"岭回归","Lasso":"Lasso回归","MLP":"多层感知机","DT":"DT决策树"
            ,"ET":"ET极限树","RF":"RF随机森林","Ada":"AdaBoost回归","GB":"梯度增强回归","bag":"Bagging回归"}

parser=argparse.ArgumentParser("Regression algorithm for SOC prediction")
parser.add_argument("--decompose",type=str,default="None",choices=["None","PCA","LDA"])
parser.add_argument("--save1",type=bool,default=False,help="whether or not to save testset_prediction_DT")
parser.add_argument("--save2",type=bool,default=True,help="whether or not to save Metrics_ALLmodels")
parser.add_argument("--file",type=str,default="test",help=" different file location to save prediction files")
parser.add_argument("--fileadd1",type=str,default="",help="change file name of Metrics_ALLmodels.csv")
parser.add_argument("--seed",type=int,default=9,help="random seed about dataset split")
parser.add_argument("--model",type=str,default="ALL",choices=["ALL","LR","KNN","SVR","Ridge","Lasso","MLP","DT","ET","RF","Ada","GB","bag"],help="choose model to solve problem")
args=parser.parse_args()

def main():
    ##读取数据
    data=pd.read_csv(path)
    ##定义x,y
    x = np.array(data[["load_ave", "load_max", "load_min", "T_ave", "T_max"]])
    # x = np.array(data[["load_max", "load_min"]])
    y = np.array(data["soc_ave"])
    # ##PCA降维
    if args.decompose=="PCA":
        x1=np.array(data[["load_ave", "load_max", "load_min","T_ave", "T_max"]])
        # x2=np.array(data[["T_ave", "T_max"]])
        pca1=PCA(n_components=4)
        # pca2=PCA(n_components=1)
        x1_new=pca1.fit_transform(x1)
        # x2_new=pca2.fit_transform(x2)
        print("PCA explained_variance_ratio: {:.2f} all".format(pca1.explained_variance_ratio_[0]))
        x= x1_new

        # x1 = np.array(data[["load_ave", "load_max", "load_min"]])
        # x2 = np.array(data[["T_ave", "T_max"]])
        # pca1 = PCA(n_components=1)
        # pca2 = PCA(n_components=1)
        # x1_new = pca1.fit_transform(x1)
        # x2_new = pca2.fit_transform(x2)
        # print("PCA explained_variance_ratio: {:.2f} (load) || {:.2f} (Temperature)".format(
        #     pca1.explained_variance_ratio_[0], pca2.explained_variance_ratio_[0]))
        # x = np.concatenate([x1_new, x2_new], axis=1)
    ###LDA有监督降维，适用于分类问题，不适用于回归问题
    # if args.decompose=="LDA":
    #     x1 = np.array(data[["load_ave", "load_max", "load_min"]])
    #     x2 = np.array(data[["T_ave", "T_max"]])
    #     lda1=LDA(n_components=1)
    #     lda2=LDA(n_components=1)
    #     x1_new=lda1.fit_transform(x1,y)
    #     x2_new=lda2.fit_transform(x2,y)
    #     print("LDA explained_variance_ratio: {:.2f} (load) || {:.2f} (Temperature)".format(
    #         lda1.explained_variance_ratio_[0], lda2.explained_variance_ratio_[0]))
    #     x = np.concatenate([x1_new, x2_new], axis=1)
    ##划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=args.seed)   ##设置不同random_state，数据集划分不同，random_state相同时，数据集划分相同
    ##归一化数据
    min_max_scaler1=preprocessing.MinMaxScaler()
    min_max_scaler2=preprocessing.MinMaxScaler()
    x_train=min_max_scaler1.fit_transform(X_train)
    y_train=min_max_scaler1.fit_transform(Y_train.reshape(-1,1))
    x_test=min_max_scaler2.fit_transform(X_test)
    y_test=min_max_scaler2.fit_transform(Y_test.reshape(-1,1))

    ####保存metrics
    metrics_list=[]
    if args.model != "ALL":
        l_model= [args.model]
    else:
        l_model = list_model
    for m in l_model:   ##遍历所有模型
        metrics_temp=[]
        args.model=m   ##指定模型
        print("model is {0}".format(m))
        ##model_prediction
        y_train_pre,y_test_pre=model_pre(args,x_train,y_train,x_test)
        ####将预测结果concat,并保存结果至csv
        ##反归一化，得到原尺度的数值(针对预测值进行反归一化，便于观察分析)
        y_train_2 = min_max_scaler1.inverse_transform(y_train_pre)
        y_test_2 = min_max_scaler2.inverse_transform(y_test_pre)
        ##保留两位小数
        y_train_2 = np.around(y_train_2, 2)
        y_test_2 = np.around(y_test_2, 2)
        data_save1=np.concatenate([Y_train.reshape(-1,1), y_train_2], axis=1)  ##拼接trainset真实值与预测结果
        data_save2=np.concatenate([Y_test.reshape(-1,1), y_test_2], axis=1)    ##拼接testset真实值与预测结果
        ##形成pandas数据
        data_save1=pd.DataFrame(data=data_save1,columns=["Truth","Prediction"])
        data_save2 = pd.DataFrame(data=data_save2, columns=["Truth", "Prediction"])
        # ##保存数据
        if not os.path.exists(r"./m_results/" + args.file):  ###新建文件夹，存储特定条件的实验结果
            os.mkdir(r"./m_results/" + args.file)
        ###保存真实值和预测值的对比   数值 表格保存
        if args.save1:
            path_save1=r"./m_results/"+args.file+"/trainset_prediction_"
            path_save2 = r"./m_results/"+args.file+"/testset_prediction_"
            path_save1=path_save1+args.model+".csv"
            path_save2 = path_save2 + args.model + ".csv"
            data_save1.to_csv(path_save1,index=False)
            data_save2.to_csv(path_save2,index=False)
        ##计算MAE,MSE
        MAE = [mean_absolute_error(Y_train.reshape(-1,1), y_train_2), mean_absolute_error(Y_test.reshape(-1,1), y_test_2)]
        MSE = [mean_squared_error(Y_train.reshape(-1,1), y_train_2), mean_squared_error(Y_test.reshape(-1,1), y_test_2)]
        R2=[r2_score(Y_train.reshape(-1,1), y_train_2),r2_score(Y_test.reshape(-1,1), y_test_2)]
        print("均方根误差 MSE(trainset,testset): {0} \n 平均绝对误差 MAE(trainset,testset): {1} \n".format(MSE, MAE))
        print("训练集R2为: {0} \n 测试集R2为: {1}\n".format(R2[0],R2[1]))
        ##添加
        metrics_temp.append(dict_model[m])
        metrics_temp.append(round(R2[0],3))
        metrics_temp.append(round(R2[1],3))
        metrics_temp.append(round(MAE[0],4))
        metrics_temp.append(round(MAE[1],4))
        metrics_temp.append(round(MSE[0],4))
        metrics_temp.append(round(MSE[1],4))
        metrics_list.append(metrics_temp)
    ###Dataframe
    list_col=["模型","R2_训练集","R2_测试集","MAE_train","MAE_test","MSE_train","MSE_test"]
    data_metrics=pd.DataFrame(metrics_list,columns=list_col)
    ##保存  12个模型的评价指标
    if args.save2:
        if args.fileadd1:   ##如果有加文件后缀
            data_metrics.to_csv(r"./m_results/"+args.file+"/Metrics_ALLmodels"+str(args.seed)+"_"+args.fileadd1+".csv",index=False)   ##保存所有模型的评价结果
        else:
            data_metrics.to_csv(
                r"./m_results/" + args.file + "/Metrics_ALLmodels" + str(args.seed)+ ".csv",
                index=False)  ##保存所有模型的评价结果
    ######检查整个流程（流程完整）
    ######绘制真实值和预测值的图片
    ######(1)采用其他普通回归模型
    ######(2)采用GBDT等更先进的机器学习回归模型
    ######(3)采用神经网络

if __name__ == '__main__':
    main()
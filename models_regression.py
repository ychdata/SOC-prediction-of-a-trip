# -*- coding:utf-8 -*-
'''
作者：ych
日期：2023年08月15日
'''
from sklearn.linear_model import LinearRegression  ##线性回归
from sklearn.neighbors import KNeighborsRegressor  ##KNN
from sklearn.svm import SVR   ##支持向量回归
from sklearn.linear_model import Ridge  ##岭回归
from sklearn.linear_model import Lasso  ##LASSO回归
from sklearn.neural_network import MLPRegressor ##感知机回归
from sklearn.tree import DecisionTreeRegressor  ##决策树回归
from sklearn.tree import ExtraTreeRegressor  ##极限树回归
from sklearn.ensemble import RandomForestRegressor  ##随机森林回归
from sklearn.ensemble import AdaBoostRegressor  ##AdaBoost回归
from sklearn.ensemble import GradientBoostingRegressor ##梯度增强回归
from sklearn.ensemble import BaggingRegressor   ##baggging回归


def model_pre(args,x_train,y_train,x_test):
    if args.model=="LR":
        ##线性回归模型
        lr=LinearRegression()
        lr.fit(x_train,y_train)
        print("线性回归的系数为:\n w={0} \n b={1} \n".format(lr.coef_,lr.intercept_))
        y_test_pre=lr.predict(x_test)
        y_train_pre=lr.predict(x_train)
    elif args.model=="KNN":
        knn=KNeighborsRegressor()
        knn.fit(x_train,y_train)
        y_train_pre=knn.predict(x_train).reshape(-1,1)
        y_test_pre=knn.predict(x_test).reshape(-1,1)
    elif args.model=="SVR":
        svr=SVR()
        svr.fit(x_train,y_train)
        y_train_pre=svr.predict(x_train).reshape(-1,1)
        y_test_pre=svr.predict(x_test).reshape(-1,1)
    elif args.model=="Ridge":
        rid=Ridge()
        rid.fit(x_train,y_train)
        y_train_pre = rid.predict(x_train).reshape(-1, 1)
        y_test_pre = rid.predict(x_test).reshape(-1, 1)
    elif args.model=="Lasso":
        lass=Lasso()
        lass.fit(x_train,y_train)
        y_train_pre = lass.predict(x_train).reshape(-1, 1)
        y_test_pre = lass.predict(x_test).reshape(-1, 1)
    elif args.model=="MLP":
        mlp=MLPRegressor()
        mlp.fit(x_train,y_train)
        y_train_pre=mlp.predict(x_train).reshape(-1,1)
        y_test_pre=mlp.predict(x_test).reshape(-1,1)
    elif args.model=="DT":
        DT=DecisionTreeRegressor()
        DT.fit(x_train,y_train)
        y_train_pre = DT.predict(x_train).reshape(-1, 1)
        y_test_pre = DT.predict(x_test).reshape(-1, 1)
    elif args.model=="ET":
        ET=ExtraTreeRegressor()
        ET.fit(x_train,y_train)
        y_train_pre = ET.predict(x_train).reshape(-1, 1)
        y_test_pre = ET.predict(x_test).reshape(-1, 1)
    elif args.model=="RF":
        rf=RandomForestRegressor()
        rf.fit(x_train,y_train)
        y_test_pre=rf.predict(x_test).reshape(-1,1)
        y_train_pre=rf.predict(x_train).reshape(-1,1)
    elif args.model=="Ada":
        Ada=AdaBoostRegressor()
        Ada.fit(x_train,y_train)
        y_train_pre = Ada.predict(x_train).reshape(-1, 1)
        y_test_pre = Ada.predict(x_test).reshape(-1, 1)
    elif args.model=="GB":
        gb=GradientBoostingRegressor()
        gb.fit(x_train,y_train)
        y_train_pre=gb.predict(x_train).reshape(-1,1)
        y_test_pre=gb.predict(x_test).reshape(-1,1)
    else:   ##Bagging回归
        bag=BaggingRegressor()
        bag.fit(x_train,y_train)
        y_test_pre=bag.predict(x_test).reshape(-1,1)
        y_train_pre=bag.predict(x_train).reshape(-1,1)

    return y_train_pre,y_test_pre

if __name__ == '__main__':
    model_pre()
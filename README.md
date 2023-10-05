# SOC-prediction-of-a-trip
Using regression model to predict SOC(state of charge) of a trip. Include 11 regression models, LR, SVR, DT, ET, RF, AdaBoost Regression, Bagging Regression and so on.

## Result
![Result of RandomForest on trainset](https://github.com/ychqingchenzhihuo/SOC-prediction-of-a-trip/blob/main/tp_train_RF.png)

![Result of RandomForest on testset](https://github.com/ychqingchenzhihuo/SOC-prediction-of-a-trip/blob/main/tp_test_RF.png)

| Model | R2_tran | R2_test | MAE_train | MAE_test | MSE_train | MSE_test |
| ---- | ----- | ------| ---- | ----- | ------|------|
| Liner Regression | 0.5 | 0.537 |	0.1683|	0.169|	0.0453|	0.0482|
| KNN	|0.739|	0.431	|0.1146|	0.1856|	0.0237|	0.0591|
| SVR |	0.675|	0.582|	0.1343|	0.1577|	0.0295|	0.0434|
| Ridge Regression |	0.475	|0.487|	0.1707|	0.176|	0.0476|	0.0533|
| 多层感知机 |	0.53	|0.554|	0.1594|	0.1649|	0.0426|	0.0463|
| DT决策树 |	1|	0.503|	0|	0.1765|	0	|0.0516|
| ET极限树 |	1	|0.429|	0|	0.1854|	0	|0.0594|
| RF随机森林	|0.95|	0.594	|0.0477|	0.1591|	0.0045|	0.0422|
| AdaBoost回归 |	0.733|	0.552|	0.1261|	0.1668|	0.0242|	0.0465|
| 梯度增强回归 |	0.883|	0.613|	0.0796|	0.1545|	0.0106|	0.0402|
| Bagging回归 |	0.933|	0.538|	0.0501|	0.1706|	0.0061|	0.048|


## Contribution
The dataset used in the code is not publicly available. But we can read or use this code to solve other regression problems. If you are interesting in the work and want to use the dataset, contact us.

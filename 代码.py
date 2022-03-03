import numpy as np
import pandas as pd
df=pd.read_csv('shuju.csv')
print(df.loc[df.a1>100].index)
print(df.loc[df.a2>100].index)
print(df.loc[df.a3>100].index)
print(df.loc[df.a4>100].index)
print(df.loc[df.a5>100].index)
print(df.loc[df.a6>100].index)
print(df.loc[df.a7>100].index)
print(df.loc[df.a8>100].index)
df=df.drop(df.index[[0,479,1963]])   #删除异常行
df.loc[df['B7']=='#NULL!','B7']=0    #将空值变为0
df.B7.value_counts()                 #查看B7列种类及其数量
df1=df[df['品牌类型'].isin([1])]
df2=df[df['品牌类型'].isin([2])]
df3=df[df['品牌类型'].isin([3])]
df1.to_csv('pinpai1.csv')
df2.to_csv('pinpai2.csv')
df3.to_csv('pinpai3.csv')
df1.describe()
df2.describe()
df3.describe()
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,confusion_matrix,make_scorer
df=pd.read_csv('pinpai1.csv')
df=df.iloc[:,3:]
print(df)
print(df['购买意愿'].value_counts())
#数据集下采样再平衡（当一个类是另一个类8倍以上采用，可调）
df_x=df.loc[df['购买意愿']==0]
df_y=df.loc[df['购买意愿']==1]
new_df_x=resample(df_x,replace=False,n_samples=len(df_y)*8)
df=pd.concat((new_df_x,df_y))
print(df['购买意愿'].value_counts())
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
sm=SMOTE()
x_train,y_train=sm.fit_resample(x_train,y_train)
print(Counter(y_train))


from sklearn.linear_model import LogisticRegression
model_logic=LogisticRegression()
model_logic.fit(x_train,y_train)
predict_logic=model_logic.predict(x_test)
print('accuracy_logic',accuracy_score(y_test,predict_logic))
print(confusion_matrix(y_test,predict_logic))
print('roc_auc',roc_auc_score(y_test,predict_logic))
print("precision",precision_score(y_test,predict_logic))
print("f1",f1_score(y_test,predict_logic))
print("recall",recall_score(y_test,predict_logic))
y_score=model_logic.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="logic_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn import tree
model_tree=tree.DecisionTreeClassifier()
model_tree.fit(x_train,y_train)
predict_tree=model_tree.predict(x_test)
print('accuracy_tree',accuracy_score(y_test,predict_tree))
print(confusion_matrix(y_test,predict_tree))
print('roc_auc',roc_auc_score(y_test,predict_tree))
print("precision",precision_score(y_test,predict_tree))
print("f1",f1_score(y_test,predict_tree))
print("recall",recall_score(y_test,predict_tree))
y_score=model_tree.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="tree_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier()
model_rf.fit(x_train,y_train)
predict_rf=model_rf.predict(x_test)
print('accuracy_rf',accuracy_score(y_test,predict_rf))
print(confusion_matrix(y_test,predict_rf))
print('roc_auc',roc_auc_score(y_test,predict_rf))
print("precision",precision_score(y_test,predict_rf))
print("f1",f1_score(y_test,predict_rf))
print("recall",recall_score(y_test,predict_rf))
y_score=model_rf.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="rf_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.naive_bayes import GaussianNB
model_bayes=GaussianNB()
model_bayes.fit(x_train,y_train)
predict_bayes=model_bayes.predict(x_test)
print('accuracy_bayes',accuracy_score(y_test,predict_bayes))
print(confusion_matrix(y_test,predict_bayes))
print('roc_auc',roc_auc_score(y_test,predict_bayes))
print("precision",precision_score(y_test,predict_bayes))
print("f1",f1_score(y_test,predict_bayes))
print("recall",recall_score(y_test,predict_bayes))
y_score=model_bayes.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="bayes_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()
model_knn.fit(x_train,y_train)
predict_knn=model_knn.predict(x_test)
print('accuracy_knn',accuracy_score(y_test,predict_knn))
print(confusion_matrix(y_test,predict_knn))
print('roc_auc',roc_auc_score(y_test,predict_knn))
print("precision",precision_score(y_test,predict_knn))
print("f1",f1_score(y_test,predict_knn))
print("recall",recall_score(y_test,predict_knn))
y_score=model_knn.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="knn_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
model_adaboost=tree.DecisionTreeClassifier()
model_adaboost=AdaBoostClassifier(base_estimator=model_adaboost,n_estimators=100)#基分类器，优化提升次数，学习率
model_adaboost.fit(x_train,y_train)
predict_adaboost=model_adaboost.predict(x_test)
print('accuracy_adaboost',accuracy_score(y_test,predict_adaboost))
print(confusion_matrix(y_test,predict_adaboost))
print('roc_auc',roc_auc_score(y_test,predict_adaboost))
print("precision",precision_score(y_test,predict_adaboost))
print("f1",f1_score(y_test,predict_adaboost))
print("recall",recall_score(y_test,predict_adaboost))
y_score=model_adaboost.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="adaboost_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
model_gbboost=GradientBoostingClassifier()
model_gbboost.fit(x_train,y_train)
predict_gbboost=model_gbboost.predict(x_test)
print('accuracy_gbboost',accuracy_score(y_test,predict_gbboost))
print(confusion_matrix(y_test,predict_gbboost))
print('roc_auc',roc_auc_score(y_test,predict_gbboost))
print("precision",precision_score(y_test,predict_gbboost))
print("f1",f1_score(y_test,predict_gbboost))
print("recall",recall_score(y_test,predict_gbboost))
y_score=model_gbboost.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="gbboost_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.ensemble import BaggingClassifier
model_bag=BaggingClassifier()
model_bag.fit(x_train,y_train)
predict_bag=model_bag.predict(x_test)
print('accuracy_bag',accuracy_score(y_test,predict_bag))
print(confusion_matrix(y_test,predict_bag))
print('roc_auc',roc_auc_score(y_test,predict_bag))
print("precision",precision_score(y_test,predict_bag))
print("f1",f1_score(y_test,predict_bag))
print("recall",recall_score(y_test,predict_bag))
y_score=model_bag.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="bag_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from xgboost import XGBClassifier
model_xgb=XGBClassifier()
model_xgb.fit(x_train,y_train)
predict_xgb=model_xgb.predict(x_test)
print('accuracy_xgb',accuracy_score(y_test,predict_xgb))
print(confusion_matrix(y_test,predict_xgb))
print('roc_auc',roc_auc_score(y_test,predict_xgb))
print("precision",precision_score(y_test,predict_xgb))
print("f1",f1_score(y_test,predict_xgb))
print("recall",recall_score(y_test,predict_xgb))
y_score=model_xgb.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="xgb_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from lightgbm import LGBMClassifier
model_lgbm=LGBMClassifier()
model_lgbm.fit(x_train,y_train)
predict_lgbm=model_lgbm.predict(x_test)
print('accuracy_lgbm',accuracy_score(y_test,predict_lgbm))
print(confusion_matrix(y_test,predict_lgbm))
print('roc_auc',roc_auc_score(y_test,predict_lgbm))
print("precision",precision_score(y_test,predict_lgbm))
print("f1",f1_score(y_test,predict_lgbm))
print("recall",recall_score(y_test,predict_lgbm))
y_score=model_lgbm.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="lgbm_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from catboost import CatBoostClassifier
model_catboost=CatBoostClassifier()
model_catboost.fit(x_train,y_train)
predict_catboost=model_catboost.predict(x_test)
print('accuracy_catboost',accuracy_score(y_test,predict_catboost))
print(confusion_matrix(y_test,predict_catboost))
print('roc_auc',roc_auc_score(y_test,predict_catboost))
print("precision",precision_score(y_test,predict_catboost))
print("f1",f1_score(y_test,predict_catboost))
print("recall",recall_score(y_test,predict_catboost))
y_score=model_catboost.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="catboost_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn import svm
model_svm=svm.SVC(kernel='linear',probability=True)
model_svm.fit(x_train,y_train)
predict_svm=model_svm.predict(x_test)
print('accuracy_svm',accuracy_score(y_test,predict_svm))
print(confusion_matrix(y_test,predict_svm))
print(roc_auc_score(y_test,predict_svm))
print("precision",precision_score(y_test,predict_svm))
print("f1",f1_score(y_test,predict_svm))
print("recall",recall_score(y_test,predict_svm))
y_score=model_svm.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="svm_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn import svm
model_svm=svm.SVC(kernel='rbf',probability=True)
model_svm.fit(x_train,y_train)
predict_svm=model_svm.predict(x_test)
print('accuracy_svm',accuracy_score(y_test,predict_svm))
print(confusion_matrix(y_test,predict_svm))
print(roc_auc_score(y_test,predict_svm))
print("precision",precision_score(y_test,predict_svm))
print("f1",f1_score(y_test,predict_svm))
print("recall",recall_score(y_test,predict_svm))
y_score=model_svm.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="svm_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn import svm
model_svm=svm.SVC(kernel='poly',probability=True)
model_svm.fit(x_train,y_train)
predict_svm=model_svm.predict(x_test)
print('accuracy_svm',accuracy_score(y_test,predict_svm))
print(confusion_matrix(y_test,predict_svm))
print(roc_auc_score(y_test,predict_svm))
print("precision",precision_score(y_test,predict_svm))
print("f1",f1_score(y_test,predict_svm))
print("recall",recall_score(y_test,predict_svm))
y_score=model_svm.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="svm_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

#MLP全连接神经网络
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import torch
import torch.nn as nn
import torch.utils.data as Data
import hiddenlayer as hl
from torchviz import make_dot
X_train_t = torch.from_numpy(x_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(x_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
train_data = Data.TensorDataset(X_train_t,y_train_t)
train_loader = Data.DataLoader(dataset = train_data,batch_size=32,num_workers = 0)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.hidden= nn.Sequential(
            nn.Linear(25,10),
            nn.ReLU(),
        )
        self.classifica = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        x= self.hidden(x)
        x= self.classifica(x)
        return x
model = model()
print(model)
x = torch.randn(1,25).requires_grad_(True)
y = model(x)
Mymlpcvis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
#Mymlpcvis.view()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
canvas1 = hl.Canvas()
print_step = 25
train_loss_all = 0
if __name__=='__main__':
    for epoch in range(15):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = model(b_x)  # MLP在训练batch上的输出
            train_loss = loss_func(output, b_y)#计算损失
            optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
            train_loss.backward()  # 损失的后向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            niter = epoch * len(train_loader) + step + 1
            train_loss_all += train_loss
            if niter % print_step == 0:
                output =model(X_test_t)
                _, pre_lab = torch.max(output, 1)
                test_accuracy = accuracy_score(y_test_t, pre_lab)
                history1.log(niter, train_loss=train_loss / niter,test_accuracy=test_accuracy)
                with canvas1:
                    canvas1.draw_plot(history1["train_loss"])
                    canvas1.draw_plot(history1["test_accuracy"])
output =model(X_test_t)
_,pre_lab = torch.max(output,1)
test_accuracy = accuracy_score(y_test_t,pre_lab)
print("test_accuracy:",test_accuracy)
print(confusion_matrix(y_test_t,pre_lab))


from sklearn.model_selection import GridSearchCV
model=RandomForestClassifier(criterion='entropy',oob_score=True)
param_grid={
    'min_samples_split':range(2,10),
    'n_estimators':[10,50,100,150],
    'max_depth':[5,10,15,20],
    'max_features':[5,10,20]
}
scorers={
    'precision_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'accuracy_score':make_scorer(accuracy_score)
}
#这个参数可以自己调，模型默认使用训练结果最优参数重新训练
refit_score='precision_score'
grid_search=GridSearchCV(model,param_grid,refit=refit_score,cv=3,return_train_score=True,scoring=scorers,n_jobs=-1)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print(grid_search.best_params_)
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pre_yes','pre_no'],index=['yes','no']))
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("f1",f1_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("roc_auc",roc_auc_score(y_test,y_pred))
y_score=grid_search.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="RandomFroest_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(max_depth=20,max_features=5,min_samples_split=2,n_estimators=50)
model_rf.fit(x_train,y_train)
predict_rf=model_rf.predict(x_test)
print('accuracy_rf',accuracy_score(y_test,predict_rf))
print(confusion_matrix(y_test,predict_rf))
print('roc_auc',roc_auc_score(y_test,predict_rf))
print("precision",precision_score(y_test,predict_rf))
print("f1",f1_score(y_test,predict_rf))
print("recall",recall_score(y_test,predict_rf))
y_score=model_rf.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="rf_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.model_selection import GridSearchCV
model=LogisticRegression()
param_grid={
    'penalty': ['l1','l2'],
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
}
scorers={
    'precision_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'accuracy_score':make_scorer(accuracy_score)
}
#这个参数可以自己调，模型默认使用训练结果最优参数重新训练
refit_score='precision_score'
grid_search=GridSearchCV(model,param_grid,refit=refit_score,cv=3,return_train_score=True,scoring=scorers,n_jobs=-1)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print(grid_search.best_params_)
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pre_yes','pre_no'],index=['yes','no']))
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("f1",f1_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("roc_auc",roc_auc_score(y_test,y_pred))
y_score=grid_search.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="RandomFroest_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from sklearn.linear_model import LogisticRegression
model_logic=LogisticRegression(C= 1, penalty='l2')
model_logic.fit(x_train,y_train)
predict_logic=model_logic.predict(x_test)
print('accuracy_logic',accuracy_score(y_test,predict_logic))
print(confusion_matrix(y_test,predict_logic))
print('roc_auc',roc_auc_score(y_test,predict_logic))
print("precision",precision_score(y_test,predict_logic))
print("f1",f1_score(y_test,predict_logic))
print("recall",recall_score(y_test,predict_logic))
y_score=model_logic.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="logic_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()
print(model_logic.coef_)#和上面的特征匹配一致

from sklearn.model_selection import GridSearchCV
model=XGBClassifier()
param_grid={
    'n_estimators': [6,10,20,50],
    'learning_rate':[0.1,0.01,0.05],
    'max_depth':range(3,10),
    'min_child_weight':range(1,6)
}
scorers={
    'precision_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'accuracy_score':make_scorer(accuracy_score)
}
#这个参数可以自己调，模型默认使用训练结果最优参数重新训练
refit_score='precision_score'
grid_search=GridSearchCV(model,param_grid,refit=refit_score,cv=3,return_train_score=True,scoring=scorers,n_jobs=-1)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print(grid_search.best_params_)
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pre_yes','pre_no'],index=['yes','no']))
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("f1",f1_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("roc_auc",roc_auc_score(y_test,y_pred))
y_score=grid_search.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="RandomFroest_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()

from xgboost import XGBClassifier
model_xgb=XGBClassifier(learning_rate= 0.01, max_depth= 3, min_child_weight= 2, n_estimators= 6)
model_xgb.fit(x_train,y_train)
predict_xgb=model_xgb.predict(x_test)
print('accuracy_xgb',accuracy_score(y_test,predict_xgb))
print(confusion_matrix(y_test,predict_xgb))
print('roc_auc',roc_auc_score(y_test,predict_xgb))
print("precision",precision_score(y_test,predict_xgb))
print("f1",f1_score(y_test,predict_xgb))
print("recall",recall_score(y_test,predict_xgb))
y_score=model_xgb.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="xgb_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()
print(model_xgb.feature_importances_)#和上面的特征匹配一致

results=grid_search.cv_results_
plt.plot(results['mean_train_accuracy_score'],label='mean_train_accuracy_score')
plt.plot(results['mean_test_accuracy_score'],label='mean_test_accuracy_score')
plt.title('model accuracy')
plt.xlabel('epoc')
plt.ylabel('accuracy')
plt.legend()
plt.show()

data=pd.read_csv('E:/Desktop/daiding.csv')
data.loc[data['B7']=='#NULL!','B7']=0    #将空值变为0
data.B7.value_counts()
#查看B7列种类及其数量
data1=data[data[data.columns[1]].isin([1])]
data2=data[data[data.columns[1]].isin([2])]
data3=data[data[data.columns[1]].isin([3])]
data1.to_csv('data1.csv')
data2.to_csv('data2.csv')
data3.to_csv('data3.csv')

data=pd.read_csv('data1.csv')
print(data)
index=data[data.columns[1]]
print(index)
data=data.iloc[:,3:-1]
data=data[['a1','a2','a3','a5','B14','B15','B16','B17']]
data

predict_logic=model_logic.predict(data)
print(predict_logic)
pred1_logic=pd.DataFrame(predict_logic,index=index,columns=['是否会购买'])
pred1_logic.to_csv('pred1_logic.csv')
pred1_logic

predict_rf=model_rf.predict(data)
print(predict_rf)
pred1_rf=pd.DataFrame(predict_rf,index=index,columns=['是否会购买'])
pred1_rf.to_csv('pred1_rf.csv')
pred1_rf

predict_xgb=model_xgb.predict(data)
print(predict_xgb)
pred1_xgb=pd.DataFrame(predict_xgb,index=index,columns=['是否会购买'])
pred1_xgb.to_csv('pred1_xgb.csv')
pred1_xgb

data=pd.read_csv('data2.csv')
print(data)
index=data[data.columns[1]]
print(index)
data=data.iloc[:,3:-1]
data=data[['a1','a2','a3','a5','B14','B15','B16','B17']]
data

predict_logic=model_logic.predict(data)
print(predict_logic)
pred2_logic=pd.DataFrame(predict_logic,index=index,columns=['是否会购买'])
pred2_logic.to_csv('pred2_logic.csv')
pred2_logic

predict_rf=model_rf.predict(data)
print(predict_rf)
pred2_rf=pd.DataFrame(predict_rf,index=index,columns=['是否会购买'])
pred2_rf.to_csv('pred2_rf.csv')
pred2_rf

predict_xgb=model_xgb.predict(data)
print(predict_xgb)
pred2_xgb=pd.DataFrame(predict_xgb,index=index,columns=['是否会购买'])
pred2_xgb.to_csv('pred2_xgb.csv')
pred2_xgb

data=pd.read_csv('data3.csv')
print(data)
index=data[data.columns[1]]
print(index)
data=data.iloc[:,3:-1]
data=data[['a1','a2','a3','a5','B14','B15','B16','B17']]
data

predict_logic=model_logic.predict(data)
print(predict_logic)
pred3_logic=pd.DataFrame(predict_logic,index=index,columns=['是否会购买'])
pred3_logic.to_csv('pred3_logic.csv')
pred3_logic

predict_rf=model_rf.predict(data)
print(predict_rf)
pred3_rf=pd.DataFrame(predict_rf,index=index,columns=['是否会购买'])
pred3_rf.to_csv('pred3_rf.csv')
pred3_rf

predict_xgb=model_xgb.predict(data)
print(predict_xgb)
pred3_xgb=pd.DataFrame(predict_xgb,index=index,columns=['是否会购买'])
pred3_xgb.to_csv('pred3_xgb.csv')
pred3_xgb
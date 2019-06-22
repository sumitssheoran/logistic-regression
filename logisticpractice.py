import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("bankinglogistic.csv")
data.info()
data=data.dropna()
data.columns
data["education"].unique()
data["education"]=np.where(data["education"]=="basic.4y","basic",data["education"])
data["education"]=np.where(data["education"]=="basic.6y","basic",data["education"])
data["education"]=np.where(data["education"]=="basic.9y","basic",data["education"])
list_object=["job","marital","education","default","housing","loan",
             "contact","month","day_of_week","poutcome"]
data=pd.get_dummies(data,columns=list_object,drop_first=True)
data.groupby("y")["age"].mean()
sns.countplot(x="y",data=data)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight="balanced",penalty="l2")
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)

from sklearn.metrics import roc_auc_score
score=roc_auc_score(y_test,y_pred_prob[:,1])
accuracy=model.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)

y_pred_prob_train=model.predict_proba(X_train)[:,1]
logreg_roc_auc=roc_auc_score(y_train,y_pred_prob_train)

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train,y_pred_prob_train)
len(thresholds)
thresholds[thresholds>0.8][-1]
plt.plot(fpr,tpr,label="area=%0.4f"%logreg_roc_auc)
plt.legend(loc="0")
from sklearn.feature_selection import RFE
model=LogisticRegression()
rfe=RFE(model,30)
rfe.fit(X_train,y_train)
rank=list(rfe.ranking_)
col=list(X_train.columns)
feature=[]
for i in range(len(col)):
    if rank[i]==1:
        feature.append(col[i])
feature    
x_train_new=pd.DataFrame()
for i in feature:
    x_train_new[i]=X_train[i]
x_test_new=pd.DataFrame()    
for i in feature:
    x_test_new[i]=X_test[i]
rfe=rfe.fit(x_train_new,y_train)
y_pred=rfe.predict(x_test_new)
y_pred_prob=rfe.predict_proba(x_test_new)[:,1]
score=roc_auc_score(y_test,y_pred_prob)

import sklearn
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

cnr=datasets.load_breast_cancer()
data=pd.read_csv("FinalStuper.csv")
datar=data[['G1','G2','studytime','absences','freetime','G3',]]

prediction='G3'
#print(cnr.feature_names)
#print(cnr.target_names)
x=np.array(datar.drop([prediction],1))

y=np.array(datar[prediction])


x_train, x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)

#classes=['malignant','benign']
clf=svm.SVC(kernel='linear',degree=15)
clf.fit(x_train,y_train)
pre=clf.predict(x_test)
accone=clf.score(x_test,y_test)
print("SVM",accone)

le=linear_model.LinearRegression()
le.fit(x_train,y_train)
pretwo=le.predict(x_test)
acctwo=le.score(x_test,y_test)
print("Linear",acctwo)

kn=KNeighborsClassifier(n_neighbors=10)
kn.fit(x_train,y_train)
prethree=kn.predict(x_test)
accthree=kn.score(x_test,y_test)
print("KNN",accthree)

tr=DecisionTreeRegressor()
tr.fit(x_train,y_train)
prefour=tr.predict(x_test)
accfour=tr.score(x_test,y_test)
print("Decision Tree",accfour)

lg=LogisticRegression()
lg.fit(x_train,y_train)
prefive=lg.predict(x_test)
accfive=lg.score(x_test,y_test)
print("Logistic Regression",accfive)

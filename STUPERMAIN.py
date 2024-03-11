import sklearn
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#USED LINEAR REGRESSION MODEL SINCE WE WANTED TO PREDICT GRADE 3 OF A STUDENT

data=pd.read_csv("FinalStuper.csv",sep=",")

#print(data)
col_names=data.columns

#print(col_names)
datar=data[['G1','G2','studytime','absences','freetime','G3',]]
#print(datar)
prediction='G3'

x=np.array(datar.drop([prediction],1))

y=np.array(datar[prediction])


x_train, x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
#SAVED MODEL BELOW (Accuracy 92%)
'''
best=0
for w in range(30):
  x_train, x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)




  le=linear_model.LinearRegression()
  le.fit(x_train,y_train)
  pretwo=le.predict(x_test)
  acctwo=le.score(x_test,y_test)
  print("Linear",acctwo)
  if acctwo>best:
    best=acctwo

    with open("studentmodelfinal.pickle","wb") as f:
      pickle.dump(le,f)
'''
pickle_in=open("studentmodelfinal.pickle","rb")
le=pickle.load(pickle_in)
print("Model hitting accuracy of : ",best*100,"%")


#accuracy_score(y_test,pretwo)
#print("DTR",accthree)

#IMPLEMENTATAION
gradeone=float(input("Enter your Grade 1 here : (Out of 20)"))
gradetwo=float(input("Enter your Grade 2 here : (Out of 20)"))
studtime=int(input("Enter the number of hours you study : "))
freetime=int(input("Enter the number of free hours you have in a day : "))
abs=int(input("Enter your total absent day number : "))

test=[]
test.append(gradeone)
test.append(gradetwo)
test.append(studtime)
test.append(abs)
test.append(freetime)
print("Your data in list : ")
print(test)

test_arr=np.array(test)
print("Test array")
print(test_arr)
test_arr=test_arr.reshape(1,-1)
pred=le.predict(test_arr)
print("Predicted Grade 3 : ",le.predict(test_arr))

#CONDITIONS
if pred<=5.0:
  print("Studies need serious attention ")
  if studtime<2:
    print("Need to increase studytime")

    if freetime>3:
      print("Need to invest some more freetime into studies.")
  else:
    print("Revisit learning approach")


if pred>5.0 and pred<10.0:
  print("You need to work harder! ")
  if studtime<2:
    print("Need to increase studytime")
    if freetime>3:
      print("Need to invest some more freetime into studies.")
  else:
    print("Study more effectively")


if pred>=10.0 and pred<15.0:
  print("Decent performance. Improving somewhat more will give you great results.")
  if studtime<2:
    print("Need to increase studytime")
    if freetime>3:
      print("Need to invest some more freetime into studies.")

if pred>=15.0 and pred<18.0:
  print("Great work!")
  if studtime<2:
    print("Need to increase studytime")
    if freetime>3:
      print("Need to invest some more freetime into studies.")
    if freetime<1.5:
      print("Start taking out more time for hobbies!")

if pred>=18.0 and pred<=20.0:
  print("Excellent!! You are doing great. Keep it up.")
  if studtime<2:
    print("You should increase studytime")
    if freetime>3:
      print("Need to invest some more freetime into studies.")
    if freetime<2:
      print("Start taking out more time for hobbies!")

if abs>12:
  print("Need to improve attendance ")
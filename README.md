# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load employee data and split it into training and testing sets.

2.Train a Decision Tree classifier using entropy as the split criterion.

3.Evaluate the model using accuracy, confusion matrix, and classification report.

4.Use the trained model to predict whether a new employee will stay or leave.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Srinithi Muthukumar
RegisterNumber:  212224240161
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
sample = pd.DataFrame(
    [[0.5,0.8,9,260,6,0,1,2]],
    columns=x.columns
)

dt.predict(sample)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['stayed','left'],
    filled=True,
    max_depth=3
)

plt.show()


```

## Output:

<img width="630" height="737" alt="image" src="https://github.com/user-attachments/assets/349cde09-0f08-4e8c-a44a-e79e3901ec24" />


<img width="1667" height="760" alt="image" src="https://github.com/user-attachments/assets/889caed7-8ce0-4c07-8bc8-f680584526e5" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

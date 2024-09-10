#Importing all the necessary liberies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn .datasets import load_diabetes


#Loading The Dataset
fraud = pd.read_csv('ccd_under_smp.csv')
fraud.head()
fraud.tail()
fraud.info()
fraud.describe()
fraud.shape


#Checking For The Null Values
fraud.isnull().sum().sum()

#Handling with null values
fraud = fraud.dropna()


#Checking For Duplicate  Value
fraud.duplicated().any()

fraud['type'].value_counts()

fraud.columns
#Visualization
plt.figure(figsize =(10,5))
fraud['type'].value_counts().plot(kind='bar')
plt.title('TRANSACTION Type Analysis')
plt.xlabel ('Type')
plt.ylabel('Count')
plt.ylim(0,5000000)
plt.show()

df = fraud.loc[fraud['isFraud']==1]
print(df.head(1))


#Create pie chart
df['type'].value_counts().plot(kind='pie',autopct='%.2f',labels=df['type'])


print(fraud.select_dtypes(include='object').nunique())

fraud[['type','nameOrig','nameDest']] = fraud[['type','nameOrig','nameDest']].astype(str)


fraud.info()


#Doing PreProcessing
from sklearn import preprocessing

for col in fraud.select_dtypes(include=['object']).columns:
  label_encoder=preprocessing.LabelEncoder()
  label_encoder.fit(fraud[col].unique())
  fraud[col]=label_encoder.transform(fraud[col])
  print(f"{col}:{fraud[col].unique()}")


#Checking Imbalanced data
print(fraud['isFraud'].value_counts())#normal 6354407, 8213

df['isFraud'].value_counts()

#data is highly imbalance  their is 2 techniques to balance the dataset
#1)UnderSampelling
normal = fraud[fraud['isFraud']==0]
fraud = fraud[fraud['isFraud']==1]
print(normal.shape)

print(fraud.shape)

normal_sample=normal.sample(n=8213)
print(normal_sample.shape)
new_data = pd.concat([normal_sample,fraud],ignore_index=True)
new_data['isFraud'].value_counts()

new_data.drop(columns = 'isFlaggedFraud', inplace = True)
new_data.drop(columns = 'step', inplace = True)


print(new_data.head(1))


#X = new_data.drop('isFraud',axis=1)
X=new_data[['type','amount','oldbalanceOrg','newbalanceOrig']]
y = new_data['isFraud']

print(X)

print(X.columns)
print(X.shape)
print(y.shape)
#Train And Test
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#Models
#1Logistic Regression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)

print("Accuracy Score of LogisticRegression With Undersampelling:" , round(accuracy_score(y_test,y_pred1)*100,2),"%")


#DecisionTree classifier
dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred2 = dt.predict(X_test)

print("Accuracy Score of DecisionTree With Undersampelling:" , round(accuracy_score(y_test,y_pred2)*100,2),"%")


#RandomForest
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)
print("Accuracy Score of RandomForest With Undersampelling:" , round(accuracy_score(y_test,y_pred3)*100,2),"%")



#Support Vector Machine
from sklearn.svm import  SVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred4 = svc.predict(X_test)

print("Accuracy Score of Svm With Undersampelling:" , round(accuracy_score(y_test,y_pred4)*100,2),"%")


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred1)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Support vector Machine With UnderSampelling:{0}'.format(svc.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()


#Confusion Matrix For Random Forest
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred1)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Random Forest With UnderSampelling:{0}'.format(rf.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()


#Save The Model
model_file = 'Credit_card_fraud_detection.pkl'
pickle.dump(rf,open(model_file,'wb'))

model = pickle.load(open(model_file,'rb'))

print(model.predict(X_test))


##  OVER SAMPELLING ##
import pandas as pd
import numpy as np
import seaborn as sns
#Load The data set
fraud = pd.read_csv('ccd_over_smp.csv')
print(fraud)
#Info about Data set
fraud.info()
#Shape of the dataset no.rows and no.of columns
print(fraud.shape)

#column name
print(fraud.columns)

#Description about data
print(fraud.describe())

#Checking For null value
print(fraud.isnull().sum().sum())


#Checking For duplicate
print(fraud.duplicated().any())


#
print(fraud['type'].value_counts())

#Visualization
import matplotlib.pyplot as plt
plt.figure(figsize =(10,5))
fraud['type'].value_counts().plot(kind='bar')
plt.title('TRANSACTION Type Analysis')
plt.xlabel ('Type')
plt.ylabel('Count')
plt.ylim(0,3000000)
plt.show()

#we lock only rows that have our target variable as 1 and compare how transcation distributed
df = fraud.loc[fraud['isFraud'] == 1]
print(df.head())

#visualazation2
df['type'].value_counts().plot(kind = 'pie',autopct = '%2f',labels=df['type'])
plt.show()



#Check unique values on object data type
print(fraud.select_dtypes(include='object').nunique)


fraud[['type','nameOrig','nameDest']] = fraud[['type','nameOrig','nameDest']].astype(str)


#Preprocessing
from sklearn import preprocessing

for col in fraud.select_dtypes(include=['object']).columns:
  label_encoder=preprocessing.LabelEncoder()
  label_encoder.fit(fraud[col].unique())
  fraud[col]=label_encoder.transform(fraud[col])
  print(f"{col}:{fraud[col].unique()}")


#Correlation matrix
plt.figure(figsize=(30,26))
sns.heatmap(fraud.corr(),fmt ='.2g', annot = True)
plt.show()



#Check for imbalance datasets
print(fraud['isFraud'].value_counts())
#normal 0:-6354407
#fraud 1:- 8213

#   oversampelling
from sklearn.utils import resample
normal = fraud[(fraud['isFraud'] == 0)]
fraud = fraud[(fraud['isFraud'] == 1)]
print(normal.shape)
fraud_sample = resample(fraud,replace=True,n_samples=6354407,random_state=0)

fraud_unsampled = pd.concat([fraud_sample,normal])
print(fraud_unsampled.head())
#Random Forest
print("random forest")




#train test spilit
from sklearn.model_selection import train_test_split
X=fraud_unsampled[['type','amount','oldbalanceOrg','newbalanceOrig']]
y = fraud_unsampled['isFraud']
print(X.shape)
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#1DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,jaccard_score,log_loss

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
print("Accuracy Score:",round(accuracy_score(y_test,y_pred)*100,2),"%")
print(precision_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(jaccard_score(y_test,y_pred))
print(log_loss(y_test,y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Reds')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Decision Tree With overSampelling:{0}'.format(dtree.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()




#2LogisticRegression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)

print("Accuracy Score of LogisticRegression With Oversampelling:" , round(accuracy_score(y_test,y_pred1)*100,2),"%")
print(precision_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(jaccard_score(y_test,y_pred1))
print(log_loss(y_test,y_pred1))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred1)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for LogisticRegression with OverSampelling:{0}'.format(log.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()


#3Support Vector Machine
from sklearn.svm import  SVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred4 = svc.predict(X_test)

print("Accuracy Score of Svm With oversampelling:" , round(accuracy_score(y_test,y_pred4)*100,2),"%")
print(precision_score(y_test,y_pred4))
print(f1_score(y_test,y_pred4))
print(recall_score(y_test,y_pred4))
print(jaccard_score(y_test,y_pred4))
print(log_loss(y_test,y_pred4))


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred4)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Support vector Machine With OverSampelling:{0}'.format(svc.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()


#4) K nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
y_pred5 = kn.predict(X_test)

print("Accuracy Score of knn With oversampelling:" , round(accuracy_score(y_test,y_pred5)*100,2),"%")
print(precision_score(y_test,y_pred5))
print(f1_score(y_test,y_pred5))
print(recall_score(y_test,y_pred5))
print(jaccard_score(y_test,y_pred5))
print(log_loss(y_test,y_pred5))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred5)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for KNN With OverSampelling:{0}'.format(kn.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()




#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)
print("Accuracy Score of RandomForest With Oversampelling:" , round(accuracy_score(y_test,y_pred3)*100,2),"%")
print(precision_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(jaccard_score(y_test,y_pred3))
print(log_loss(y_test,y_pred3))


#Confusion Matrix For Random Forest
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred3)

plt.figure(figsize=(10,10))
sns.heatmap(data=cm,linewidths=5, fmt = '.1f', annot = True, cmap = 'Greens')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Random Forest With overSampelling:{0}'.format(rf.score(X_test,y_test))
plt.title(all_sample_title,size = 15)
plt.tight_layout()
plt.show()



#CompareAll 5 models
final_data = pd.DataFrame({'Models':['LR','DT','RF','KN','SVM'],
              "ACC": [accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred)*100,
                     accuracy_score(y_test,y_pred3)*100,
                     accuracy_score(y_test,y_pred4)*100,
                     accuracy_score(y_test,y_pred5)*100
                    ]})
final_data

sns.barplot(final_data['Models'],final_data['ACC'])
plt.show()




#Save The Model
import pickle
model_file = 'Credit_card_fraud_detection.pkl'
pickle.dump(dtree,open(model_file,'wb'))

model = pickle.load(open(model_file,'rb'))

print(model.predict(X_test))




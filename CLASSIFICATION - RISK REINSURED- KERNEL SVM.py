#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('train.csv')
X= dataset.iloc[:,:9].values
y = dataset.iloc[:,9].values

#categorical data handling
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

l=[1,3,4,5,6,7]
count = 0

for j in l :
     X[:,j] = labelencoder.fit_transform(X[:,j])

y = labelencoder.fit_transform(y) 
     
#dummy variable trap    
for i,j in enumerate(l) :
    onc = OneHotEncoder(categorical_features = [count + j - (2*i)])
    onc.fit(X)
    count = count+ int(onc.n_values_)
    X = onc.fit_transform(X).toarray()
    X= X[:,1:]
    
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0) 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)

#predicting
y_pred =classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


























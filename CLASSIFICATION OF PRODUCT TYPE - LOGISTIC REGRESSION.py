#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[0,1,2,3,4,5,7]].values
y = dataset.iloc[:,6].values

#categorical data handling
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder = LabelEncoder()

X[:,1]=  labelencoder.fit_transform(X[:,1])
X[:,3]=  labelencoder.fit_transform(X[:,3])
X[:,4]=  labelencoder.fit_transform(X[:,4])
X[:,5]=  labelencoder.fit_transform(X[:,5])
X[:,6]=  labelencoder.fit_transform(X[:,6])
y = labelencoder.fit_transform(y)

#dummy variable trap

l = [1,3,4,5,6]
count = 0 

for i,j in enumerate(l) :
    onc = OneHotEncoder(categorical_features =[count + j -(2*i)])
    onc.fit(X)
    count = count + int(onc.n_values_)
    X = onc.fit_transform(X).toarray()
    X = X[:,1:]

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#dimentionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as  LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

#classification model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#predicting results
y_pred = classifier.predict(X_test)

#k fold cross vaidation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X= X_train,y = y_train,cv=5)


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green','blue','pink','yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue','pink','yellow'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 


























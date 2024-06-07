# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as ske
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import utils
import joblib
import sys
import pickle
sys.modules['sklearn.externals.joblib'] = joblib



data = pd.read_csv('data.csv',sep="|")
data.head()


data.isnull().sum()

colomuns = ["LoaderFlags","NumberOfRvaAndSizes","SectionsNb","SectionsMeanEntropy","SectionsMinEntropy","SectionsMaxEntropy","SectionsMeanRawsize","SectionMaxRawsize","SectionsMeanVirtualsize","SectionsMinVirtualsize","SectionMaxVirtualsize","ImportsNbDLL","ImportsNb","ImportsNbOrdinal","ExportNb","ResourcesNb","ResourcesMeanEntropy","ResourcesMinEntropy","ResourcesMaxEntropy","ResourcesMeanSize","ResourcesMinSize","ResourcesMaxSize","LoadConfigurationSize","VersionInformationSize","legitimate"]
for c in colomuns:  
  m=round(data[c].mean(),2)
  data= data.fillna(m)

X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

data.dtypes

#sns.countplot(x='legitimate', data=data);

ex = ExtraTreesClassifier()
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

fsel = ex.fit(X,y_transformed)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.2)

features = []

print('%i features identified as important:' % nb_features)

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

rd = RandomForestClassifier()
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_train)
rd.fit(X_train, y_transformed)
pred = rd.predict(X_test)
score = rd.score(X_test, y_test)
accuracy = accuracy_score(y_test,pred)
print("Accuracy:", accuracy)

# Display confusion matrix
cf_matrix = confusion_matrix(y_test,pred)
print("Confusion Matrix:")
print(cf_matrix)

plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.2%')
plt.show()
print('\033[31m###################- End -###################\033[0m')



# Save the algorithm and the feature list for later predictions
print('Saving algorithm and feature list in classifier directory...')
joblib.dump(rd, 'classifier.pkl')
open('features.pkl', 'bw').write(pickle.dumps(features))
print('Saved')

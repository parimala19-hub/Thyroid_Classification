# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:14:31 2024

@author: SATHVIK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


data = pd.read_csv(r'C:\Users\SATHVIK\OneDrive\Desktop\smartinternz\Thyroid Classification\5. project Executable Files\Data\thyroidDF.csv')

data

data.head()

data.isnull()

data.isnull().sum()


# Drop redundant attributes and modify the original dataframe
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'referral_source', 'patient_id'], axis=1, inplace=True)


# Remapping target values to diagnostic group
diagnoses = {
    'A': 'hyperthyroid conditions', 'B': 'hyperthyroid conditions',
    'C': 'hyperthyroid conditions', 'D': 'hyperthyroid conditions',
    'E': 'hypothyroid conditions', 'F': 'hypothyroid conditions',
    'G': 'hypothyroid conditions', 'H': 'hypothyroid conditions',
    'I': 'binding protein', 'J': 'binding protein',
    'K': 'general health', 'L': 'replacement therapy',
    'M': 'replacement therapy', 'N': 'replacement therapy',
    'O': 'antithyroid treatment', 'P': 'antithyroid treatment',
    'Q': 'antithyroid treatment', 'R': 'miscellaneous',
    'S': 'miscellaneous', 'T': 'miscellaneous'
}

data['target'] = data['target'].map(diagnoses)
data.dropna(subset=['target'], inplace=True)

data['target'].value_counts()


data.describe()

data.info()

print(data[data.age > 100])

#changing age of observation with(age>100) to null
data['age']=np.where((data.age>100), np.nan, data.age)

data


# Split the data
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

x

y

x['sex'].unique()

x['sex'].replace(np.nan, 'F', inplace=True)

x['sex'].value_counts()


# Converting the data
x['age'] = x['age'].astype('float')
x['TSH'] = x['TSH'].astype('float')
x['T3'] = x['T3'].astype('float')
x['TT4'] = x['TT4'].astype('float')
x['T4U'] = x['T4U'].astype('float')
x['FTI'] = x['FTI'].astype('float')
x['TBG'] = x['TBG'].astype('float')


x.info()

# Ordinal encoding for categorical features
ordinal_encoder = OrdinalEncoder(dtype='int64')
x[x.columns[1:16]] = ordinal_encoder.fit_transform(x.iloc[:, 1:16])
x.fillna(0, inplace=True)

x

# Label encoding for the target variable
label_encoder = LabelEncoder()
y = pd.DataFrame(label_encoder.fit_transform(y), columns=['target'])

y

# checking correlation using Heatmap
import seaborn as sns
corrmat = x.corr()
f, ax = plt.subplots (figsize = (9, 8))
sns.heatmap (corrmat, ax = ax, cmap = "YlGnBu", linewidths = 0.1)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# Scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Handling imbalanced data
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0, k_neighbors=1)
x_bal, y_bal = os.fit_resample(x_train, y_train)
x_test_bal, y_test_bal = os.fit_resample(x_test, y_test)

print(y_train.value_counts())

x_bal

# Convert arrays to dataframes
columns = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
x_train_bal = pd.DataFrame(x_bal, columns=columns)
y_train_bal = pd.DataFrame(y_bal, columns=['target'])
x_test_bal = pd.DataFrame(x_test_bal, columns=columns)
y_test_bal = pd.DataFrame(y_test_bal, columns=['target'])

# Random Forest Classifier Model
rfr = RandomForestClassifier().fit(x_bal, y_bal.values.ravel())
y_pred = rfr.predict(x_test_bal)
print(classification_report(y_test_bal, y_pred))

print(x_bal.shape, y_bal.shape, x_test_bal.shape, y_test_bal.shape)

from sklearn.metrics import accuracy_score
test_score=accuracy_score(y_test_bal,y_pred)

test_score

train_score = accuracy_score(y_bal,rfr.predict(x_bal))
train_score


# Feature importance
from sklearn.inspection import permutation_importance
results = permutation_importance(rfr, x_bal, y_bal, scoring='accuracy')
feature_importance = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
importance = results.importances_mean
importance = np.sort(importance)
for i, v in enumerate(importance):
    i = feature_importance[i]
    print('Feature: {:<20} Score: {}'.format(i, v))
    


plt.figure(figsize=(10, 10))
plt.bar(feature_importance, importance.astype(float))
plt.xticks(rotation=30, ha='right')
plt.show()

x.head()


# Convert x_bal back to a DataFrame if necessary
if isinstance(x_bal, np.ndarray):
    x_bal = pd.DataFrame(x_bal, columns=columns)  # Assuming 'columns' contains the column names

# Now you can drop the columns
x_bal.drop(['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment'], axis=1, inplace=True)
# Drop the specified columns from the dataframe (assuming x_test_bal is still a DataFrame)
x_test_bal.drop(['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment'], axis=1, inplace=True)


x_bal.head()

x_bal_filtered = x_bal[['goitre', 'tumor' , 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']]
x_test_bal_filtered = x_test_bal[['goitre', 'tumor' , 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']]


x_bal_filtered.head()


from sklearn.ensemble import RandomForestClassifier

RFclassifier = RandomForestClassifier(max_leaf_nodes=30)
RFclassifier.fit(x_train, y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix # Import confusion_matrix

RFclassifier = RandomForestClassifier(max_leaf_nodes=30)
RFclassifier.fit(x_train, y_train)

y_pred = RFclassifier.predict(x_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))


# HyperParameter Tuning for RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, None]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf, cv=5)
grid_rf.fit(x_train, y_train)

print("Best parameters for Random Forest:", grid_rf.best_params_)

from sklearn.model_selection import GridSearchCV
rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, None]
}
RFclassifier = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(RFclassifier, rf, cv=5)
grid_rf.fit(x_train, y_train)
print("Best parameters for Random Forest:", grid_rf.best_params_)
y_pred = grid_rf.predict(x_test)
print(classification_report(y_test, y_pred))


print(confusion_matrix(y_test, y_pred))


RFAcc = accuracy_score(y_pred,y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))




from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
xgb = XGBClassifier()
xgb.fit(x_train, y_train_encoded)


y_test_encoded = le.transform(y_test)
y_pred = xgb.predict(x_test)
print(classification_report(y_test_encoded, y_pred))


from sklearn.model_selection import GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [100, 150, 200],
}
grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5)
grid_search.fit(x_train, y_train_encoded)
best_params = grid_search.best_params_
print("Best parameters:", best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)


XGBAcc = accuracy_score(y_test_encoded, y_pred)
print('XGB accuracy is: {:.2f}%'.format(XGBAcc*100))


from sklearn.svm import SVC
SVCclassifier = SVC(kernel='linear', max_iter=251)
SVCclassifier.fit(x_train, y_train)
# Hyperparametric Tuning for SVC model
svc_params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto']
}

grid_svc = GridSearchCV(SVC(), svc_params, cv=5)
grid_svc.fit(x_train, y_train)

print("Best parameters for SVC:", grid_svc.best_params_)

y_pred = SVCclassifier.predict(x_test)
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

#Evaluating Performance Of The svc Model Using GridSearch CV

from sklearn.model_selection import GridSearchCV
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))


models = pd.DataFrame({
    'Model' : ['Random Forest Classifier','SVC Model','XGB classifier'],
    'Score' : [RFAcc,SVCAcc,XGBAcc]
    })
models.sort_values(by = 'Score', ascending = False)


models = pd.DataFrame({
    'Model': ['Random Forest Classifier', 'SVC Model', 'XGB classifier'],
    'Score': [RFAcc, SVCAcc, XGBAcc]
})


models_sorted = models.sort_values(by='Score', ascending=False)
models_sorted

best_model_name = models_sorted.iloc[0, 0]
if best_model_name == 'Random Forest Classifier':
    best_model = grid_rf.best_estimator_
elif best_model_name == 'SVC Model':
    best_model = grid_svc.best_estimator_
elif best_model_name == 'XGB classifier':
    best_model = grid_search.best_estimator_
    
    
features = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000000,0.0,0.0,1.00,0.0,40.0]])
print(label_encoder.inverse_transform(xgb.predict(features)))

data['target'].unique()

y['target'].unique()


import pickle

pickle.dump(best_model, open('thyroid_1_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb')) 
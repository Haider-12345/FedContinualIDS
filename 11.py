import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv')
df = df.drop(columns=['originh', 'originp', 'responh', 'responp','traffic_category'])

y = df['Label']
X = df.drop(['Label'], axis=1)
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)
print(selected_features)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# check for duplicated features in the training set
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)
 
    col_1 = X_train.columns[i]
 
    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
            
print(len(duplicated_feat))
'''
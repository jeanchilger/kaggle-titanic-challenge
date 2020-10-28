import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Initial EDA:
# print(raw_data.columns)
# print(raw_data.isnull().sum())
# print(test_data.isnull().sum())

##########################
# Data preprocessing
##########################
used_features = [
    'Pclass', 
    # 'Name',
    'Sex', 
    'Age', 
    'SibSp',
    'Parch', 
    # 'Ticket', 
    'Fare', 
    # 'Cabin', 
    'Embarked'
]

print(raw_data.Sex)

for example in raw_data:
    pass

raw_y = raw_data.Survived
raw_X = raw_data[used_features]

X_train, X_test, y_train, y_test = train_test_split(
    raw_X, 
    raw_y, 
    test_size=0.2, 
    random_state=1
)

##########################
# Model definition
##########################
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

##########################
# Model evaluation
##########################
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

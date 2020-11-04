import csv
import pandas as pd
import random

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.linear_model import SGDClassifier, LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neural_network import MLPClassifier

raw_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

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

label_encoder = LabelEncoder()
for dataset in [raw_data, test_data]:
    # Imput missing values
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    # Remove remaining rows with nulls
    dataset.dropna(subset=['Embarked'], axis=0, inplace=True) 

    # Encode features to discrete values
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['Embarked'] = label_encoder.fit_transform(dataset['Embarked'])

    # TODO: infer new features?

# Split data into train and test
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
def generate_svm_models(n_models=4):
    '''
        Creates and returns SVM models.
        The parameters of the created models are
        randomic choosen.
    '''
    models = []

    for i in range(n_models):
        models.append(
            SVC(
                C=random.uniform(0.9, 1.7),
                kernel=random.choice(['linear', 'poly', 'rbf', 'sigmoid']),
                # degree=random.randint(3, 4),
                gamma=random.choice(['scale', 'auto'])
            )
        )

    return models

model_titles = [
    'Random Forest',
    'Gradient Boost',
    'Ada Boost',

    'Multi-layer Perceptron',

    'Gaussian NB',
    'Bernoulli NB',

    'Logistic Regression',
    'SGD Classification',

    'SVM 1',
    'SVM 2',
    'SVM 3',
]

models = [
    RandomForestClassifier(
        n_estimators=100,
        random_state=1
    ),

    GradientBoostingClassifier(
        n_estimators=100,
        random_state=1
    ),

    AdaBoostClassifier(
        n_estimators=50,
        random_state=1
    ),

    MLPClassifier(
        max_iter=300
    ),

    GaussianNB(),

    BernoulliNB(),

    LogisticRegressionCV(),

    SGDClassifier(),
]

for svc_model in generate_svm_models(3):
    models.append(svc_model) 

models_ensemble = VotingClassifier(
    estimators=[tuple(pair) for pair in zip(model_titles, models)]
)

models_ensemble.fit(X_train, y_train)

##########################
# Model evaluation
##########################
y_pred = models_ensemble.predict(X_test)

print('Ensemble Model')
print(classification_report(y_test, y_pred))
print()

##########################
# Submission
##########################
with open('submission.csv', 'w') as submission_file:
    writer = csv.writer(submission_file)
    indexes = test_data['PassengerId']
    
    writer.writerow(['PassengerId', 'Survived'])

    for i in range(len(y_pred)):
        writer.writerow([indexes[i], y_pred[i]])


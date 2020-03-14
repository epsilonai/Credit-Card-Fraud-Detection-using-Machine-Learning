import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from joblib import dump, load
from sklearn.pipeline import Pipeline
import os

from CreditCardFraudModel.preprocess import preprocessing

def RunModel():
    file_path = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
    dataset = pd.read_csv(file_path)

    features = dataset.iloc[:,:-1]

    goal = dataset.iloc[:,-1]

    preproces = preprocessing()

    preproces.fit(features)
    new_features = preproces.transform(features)

    train_set, test_set, goal_train, goal_test = train_test_split(new_features,goal,test_size =0.2,random_state=0)
    print('data splitted.....')

    knn = KNeighborsClassifier(n_neighbors = 5)
    ksvc_final = SVC(kernel ='rbf', random_state = 0 , C = 1)
    lsvc_final = LinearSVC(C = 1, loss = "hinge", penalty = "l2")
    dt_final = DecisionTreeClassifier(criterion = "gini", min_samples_leaf=8, max_depth=10)
    lda_final = LDA(n_components=1)

    eclf = VotingClassifier(estimators=[('knn', knn), ('ksvc', ksvc_final), ('lsvc', lsvc_final), ('dt', dt_final), ('lda', lda_final)], voting='hard')
    # eclf.fit(train_set,goal_train)
    # goal_predict = eclf.predict(test_set)
    # print(classification_report(goal_test,goal_predict))

    pipe = Pipeline([('preprocessor', preprocessing()), ('classifier', eclf)])

    pipe.fit(train_set,goal_train)
    print('fitting done.....')

    dump(pipe, "creditcardfraudmodel.pkl", True)
    print('dumping complete.....')
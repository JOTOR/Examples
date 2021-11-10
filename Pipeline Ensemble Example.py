import pandas as pd
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

svc= make_pipeline(StandardScaler(), LinearSVC(C=0.1, class_weight='balanced', random_state=1234))
rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=5, max_depth=3, class_weight='balanced', random_state=1234))
dt = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=1234))

estimators=[('svc', svc), ('rf', rf), ('dt', dt)]

vot_c = VotingClassifier(estimators=estimators)

stk = StackingClassifier(estimators=estimators, 
                        final_estimator=LogisticRegression(C=0.1, class_weight='balanced', random_state=1234), cv=5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1234)

svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)

rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)

dt.fit(X_train, y_train)
dt.score(X_train, y_train)
dt.score(X_test, y_test)

vot_c.fit(X_train, y_train)
vot_c.score(X_train, y_train)
vot_c.score(X_test, y_test)

stk.fit(X_train, y_train)
stk.score(X_train, y_train)
stk.score(X_test, y_test)

svc.score(X_test, y_test)
rf.score(X_test, y_test)
dt.score(X_test, y_test)
vot_c.score(X_test, y_test)
stk.score(X_test, y_test)
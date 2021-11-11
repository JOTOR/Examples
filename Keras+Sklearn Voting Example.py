## Loading required libraries
import tensorflow
tensorflow.random.set_seed(1234)
import numpy as np
np.random.seed(1234)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Loading data
X, y = load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Building Keras Classifier
def model_fn():
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1], ), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

k = KerasClassifier(build_fn=model_fn, epochs=20, batch_size=32, validation_split = 0.2)

# Yes it is possible to include this estimator within a pipeline
pla = make_pipeline(StandardScaler(), k)
pla.fit(X_train, y_train)
pla.get_params()

#Second model
plb = DecisionTreeClassifier(max_depth=3, random_state=1234)
plb.fit(X_train, y_train)

#Third model
plc = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', random_state=1234))
plc.fit(X_train, y_train)

#Scores - Sanity Check the 3 models are working as expected
pla.score(X_test, y_test)
plb.score(X_test, y_test)
plc.score(X_test, y_test)

## Unfortunately the VotingClassifier does not accept the KerasClassifier as an estimator, the 
## following lines are a work around for a VotingClassifier using a "hard" strategy

res = np.hstack((pla.predict(X_test).reshape(-1, 1), 
                 plb.predict(X_test).reshape(-1, 1),
                 plc.predict(X_test).reshape(-1, 1)))
res

pred_class = [np.bincount(res[i]).argmax() for i in np.arange(len(res))]
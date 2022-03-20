from MLFeatureSelection import importance_selection
from sklearn.tree import DecisionTreeClassifier
from ml_classifier.svc import get_config
import json
import numpy as np
from pkg_resources import resource_filename
import pandas as pd
from sklearn.model_selection import train_test_split


def lossfunction(y_pred, y_test):
    """define your own loss function with y_pred and y_test
    return score
    """
    return np.mean(y_pred == y_test)


def validate(X, y, features, clf, lossfunction):
    """define your own validation function with 5 parameters
    input as X, y, features, clf, lossfunction
    clf is set by SetClassifier()
    lossfunction is import earlier
    features will be generate automatically
    function return score and trained classfier
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("what are the features", features)
    print(X_train[features])
    clf.fit(X_train[features], y_train)
    y_pred = clf.predict(X_test[features])
    score = lossfunction(y_pred, y_test)
    return score, clf


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    feature_path = resource_filename(__name__, config['feature_file_path']['path'])
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()
    Y = [0] * len(neg_lines) + [1] * len(pos_lines)

    X = []
    with open(feature_path) as f:
        lines = f.readlines()
    for i in lines:
        obj = json.loads(i)
        feature_vec = np.array(obj['feature'])
        feature = np.asarray([feature_vec])
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)
    df = pd.DataFrame(np.concatenate(X))
    df.columns = [str(i) for i in range(0, 135)]
    df['outcome'] = pd.Series(Y)

    print(df)

    sf = importance_selection.Select()
    sf.ImportDF(df, label='0')  # import dataframe and label
    sf.ImportLossFunction(lossfunction, direction='ascend')  # import loss function and optimize direction
    sf.InitialFeatures(features=[str(i) for i in range(0, 134)])  # initial features, input
    sf.SelectRemoveMode(batch=2)
    sf.clf = DecisionTreeClassifier(max_depth=6, random_state=0)
    sf.SetLogFile('record.log')  # log file
    sf.run(validate)  # run with validation function, return best features combination

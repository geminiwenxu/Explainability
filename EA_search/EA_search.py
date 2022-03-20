import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pkg_resources import resource_filename
import yaml
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def ea(neg_path, pos_path):
    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()

    Y = [0] * len(neg_lines) + [1] * len(pos_lines)
    X = []
    for i in neg_lines:
        obj = json.loads(i)
        feature = np.array(obj['feature'])
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)
    for i in pos_lines:
        obj = json.loads(i)
        feature = np.array(obj['feature'])
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    clf = DecisionTreeClassifier(max_depth=6, random_state=0)
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf = SVC(gamma='auto', cache_size=500)
    evolved_estimator = GAFeatureSelectionCV(estimator=clf, cv=3, scoring=None, population_size=10, generations=80,
                                             crossover_probability=0.8, mutation_probability=0.2, tournament_size=3,
                                             elitism=True, max_features=None, verbose=True, keep_top_k=1,
                                             criteria='max', algorithm='eaMuPlusLambda', refit=True, n_jobs=1,
                                             pre_dispatch='2*n_jobs', return_train_score=False,
                                             log_config=None)
    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.best_features_
    print(features)
    y_predict_ga = evolved_estimator.predict(X_test[:, features])
    print(accuracy_score(y_test, y_predict_ga))
    print(f1_score(y_test, y_predict_ga, average=None))


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    ea(neg_path, pos_path)

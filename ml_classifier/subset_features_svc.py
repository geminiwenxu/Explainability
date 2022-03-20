import json
from itertools import combinations
from pkg_resources import resource_filename
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main(set=2):
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    feature_path = resource_filename(__name__, config['feature_file_path']['path'])
    print(neg_path)

    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()
    Y = [0] * len(neg_lines) + [1] * len(pos_lines)

    with open(feature_path) as f:
        lines = f.readlines()
    for j in range(1, set):
        array = np.arange(135)
        num_combinations = len(list(combinations(array, j)))
        print('j', j)
        print('number of combinations: ', num_combinations)
        for q in range(0, num_combinations):
            print('q', q)
            X = []
            for i in lines:
                obj = json.loads(i)
                feature_vec = np.array(obj['feature'])
                features = list(combinations(feature_vec, j))
                feature = np.asarray(features[q])
                feature = np.nan_to_num(feature.astype(np.float32))
                X.append(feature)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            clf = SVC(gamma='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            class_names = ['correct_classified', 'misclassified']
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            repdf = pd.DataFrame(report_dict).round(2).transpose()
            repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
            save_path = resource_filename(__name__, config['svm_save_path']['path'])
            repdf.to_csv(
                save_path + "{} combination on {} feature.csv".format(j, q),
                index=False)


if __name__ == "__main__":
    main()

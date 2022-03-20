import json
from itertools import combinations
from pkg_resources import resource_filename
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    feature_path = resource_filename(__name__, config['feature_file_path']['path'])
    feature_names = config['feature_names']

    decision_features = ['depH', 'NDW', 'Q', 'dpd', 'advpd', 'RR', 'preppd', 'adjpd', 'btac1', 'G', 'imbG', 'wG',
                         'DDEmu', 'btH', 'lmbd', 'ASL', 'btadc10', 'npd', 'RRR', 'btadc10', 'Lradc', 'btrac', 'btac4',
                         'deprac', 'btadc2', 'btadc10', 'UG', 'btac1', 'btac3', 'btac2', 'btac10', 'alpha', 'Lradc',
                         'cG', 'btrac', 'btadc9', 'btac4', 'DDEradc', 'deprac', 'depG']

    lexical_features = ['adjpd', 'advpd', 'apd', 'dpd', 'ipd', 'npd', 'NDW', 'ppd', 'preppd', 'vpd']

    syntactic_features = ['LDEmu', 'depmu', 'MDDmu', 'DDEmu', 'TCImu', 'imbmu', 'Lmu', 'Wmu', 'wmu', 'lmu', 'cmu',
                          'LDEG', 'depG', 'MDDG', 'DDEG', 'TCIG', 'imbG', 'LG', 'WG', 'wG', 'lG', 'cG',
                          'LDEH', 'depH', 'MDDH', 'DDEH', 'TCIH', 'imbH', 'LH', 'WH', 'wH', 'lH', 'cH',
                          'LDErac', 'deprac', 'MDDrac', 'DDErac', 'TCIrac', 'imbrac', 'Lrac', 'Wrac', 'wrac', 'lrac',
                          'crac', 'LDEadc', 'depadc', 'MDDadc', 'DDEadc', 'TCIadc', 'imbadc', 'Ladc', 'Wadc', 'wadc',
                          'ladc', 'cadc', 'LDEradc', 'depradc', 'MDDradc', 'DDEradc', 'TCIradc', 'imbradc', 'Lradc',
                          'Wradc', 'wradc', 'lradc', 'cradc', 'LDEadtw', 'depadtw', 'MDDadtw', 'DDEadtw', 'TCIadtw',
                          'imbadtw', 'Ladtw', 'Wadtw', 'wadtw', 'ladtw', 'cadtw']

    bert_features = ['btH', 'btlH', 'btsH', 'btlsH', 'bth', 'btdfa', 'btly', 'btrac', 'btradc', 'btac1', 'btac2',
                     'btac3', 'btac4', 'btac5', 'btac6', 'btac7', 'btac8', 'btac9', 'btac10', 'btadc1', 'btadc2',
                     'btadc3', 'btadc4', 'btadc5', 'btadc6', 'btadc7', 'btadc8', 'btadc9', 'btadc10']

    semantic_features = ['A', 'alpha', 'ATL', 'ASL', 'L', 'H', 'G', 'hl', 'h', 'lmbd', 'Q', 'R1', 'RR', 'RRR', 'stc',
                         'tc', 'ttr', 'UG', 'VD']

    best_combination = [0, 2, 3, 6, 8, 10, 16, 18, 19, 21, 22, 26, 32, 34, 35, 45, 48, 52, 87, 106, 113, 114,
                        115, 117, 118, 124, 126, 127]
    ls_names =[]
    for i in best_combination:
        name = feature_names[i]
        ls_names.append(name)
    print(ls_names)
    ls_index = []
    for i in decision_features:  # using different features
        index = feature_names.index(i)
        ls_index.append(index)

    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()
    Y = [0] * len(neg_lines) + [1] * len(pos_lines)

    with open(feature_path) as f:
        lines = f.readlines()

    X = []
    for i in lines:
        obj = json.loads(i)
        feature_vec = np.array(obj['feature'])
        features = [feature_vec[i] for i in
                    ls_index]  # using all useful features from decision tree or single feature list skip this line
        feature = np.asarray(features)  # feature_vec instead of features
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = SVC(gamma='auto', cache_size=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    class_names = ['correct_classified', 'misclassified']
    print(classification_report(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    repdf = pd.DataFrame(report_dict).round(2).transpose()
    repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    save_path = resource_filename(__name__, config['svm_save_path']['path'])
    repdf.to_csv(save_path + "best combination of features.csv", index=False)


if __name__ == "__main__":
    main()

import json
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pkg_resources import resource_filename
import yaml
from genetic_selection import GeneticSelectionCV


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def prepare_data(neg_path, pos_path):
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
    return X_train, X_test, y_train, y_test


def decision_tree(X_train, y_train, X_test):
    clf = DecisionTreeClassifier(max_depth=6, random_state=0)
    model = GeneticSelectionCV(clf, cv=5, verbose=0,
                               scoring="accuracy", max_features=100,
                               n_population=100, crossover_proba=0.5,
                               mutation_proba=0.2, n_generations=50,
                               crossover_independent_proba=0.5,
                               mutation_independent_proba=0.04,
                               tournament_size=3, n_gen_no_change=10,
                               caching=True, n_jobs=-1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=['adjpd', 'A', 'advpd', 'alpha', 'apd', 'ATL', 'ASL', 'L', 'dpd', 'H',
                                                   'G', 'hl', 'h', 'ipd', 'npd', 'lmbd', 'NDW', 'ppd', 'preppd', 'Q',
                                                   'R1', 'RR', 'RRR', 'stc', 'tc', 'ttr', 'UG', 'VD', 'vpd', 'LDEmu',
                                                   'depmu', 'MDDmu', 'DDEmu', 'TCImu', 'imbmu', 'Lmu', 'Wmu', 'wmu',
                                                   'lmu', 'cmu', 'LDEG', 'depG', 'MDDG', 'DDEG', 'TCIG', 'imbG', 'LG',
                                                   'WG', 'wG', 'lG', 'cG', 'LDEH', 'depH', 'MDDH', 'DDEH', 'TCIH',
                                                   'imbH', 'LH', 'WH', 'wH', 'lH', 'cH', 'LDErac', 'deprac', 'MDDrac',
                                                   'DDErac', 'TCIrac', 'imbrac', 'Lrac', 'Wrac', 'wrac', 'lrac', 'crac',
                                                   'LDEadc', 'depadc', 'MDDadc', 'DDEadc', 'TCIadc', 'imbadc', 'Ladc',
                                                   'Wadc', 'wadc', 'ladc', 'cadc', 'LDEradc', 'depradc', 'MDDradc',
                                                   'DDEradc', 'TCIradc', 'imbradc', 'Lradc', 'Wradc', 'wradc', 'lradc',
                                                   'cradc', 'LDEadtw', 'depadtw', 'MDDadtw', 'DDEadtw', 'TCIadtw',
                                                   'imbadtw', 'Ladtw', 'Wadtw', 'wadtw', 'ladtw', 'cadtw', 'btH',
                                                   'btlH', 'btsH', 'btlsH', 'bth', 'btdfa', 'btly', 'btrac', 'btradc',
                                                   'btac1', 'btac2', 'btac3', 'btac4', 'btac5', 'btac6', 'btac7',
                                                   'btac8', 'btac9', 'btac10', 'btadc1', 'btadc2', 'btadc3', 'btadc4',
                                                   'btadc5', 'btadc6', 'btadc7', 'btadc8', 'btadc9', 'btadc10'],
                                    class_names=['correct', 'misclassified'],
                                    filled=True)
    # graph = graphviz.Source(dot_data, format="png")
    # graph.render("decision_tree")
    return y_pred


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.subplots_adjust(bottom=.25, left=.25)
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.savefig('confusion matrix')


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])

    X_train, X_test, y_train, y_test = prepare_data(neg_path, pos_path)

    y_pred = decision_tree(X_train, y_train, X_test)
    class_names = ['correct_classified', 'misclassified']
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)

    print(classification_report(y_test, y_pred, target_names=class_names))

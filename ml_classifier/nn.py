from keras.models import Sequential
from keras.layers import Dense
from pkg_resources import resource_filename
import yaml
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout
from keras.constraints import maxnorm


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def prepare_data(neg_path, pos_path):
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
    ls_index = []
    for i in bert_features:  # using different features
        index = feature_names.index(i)
        ls_index.append(index)
    print(ls_index)
    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()

    Y = [0] * len(neg_lines) + [1] * len(pos_lines)
    X = []
    for i in neg_lines:
        obj = json.loads(i)
        feature_vec = np.array(obj['feature'])
        features = [feature_vec[i] for i in
                    ls_index]  # using all useful features from decision tree or single feature list skip this line
        feature = np.asarray(features)  # feature_vec instead of features
        feature = np.nan_to_num(feature.astype(np.float32))
        for i in range(len(feature)):
            if feature[i] > 205:
                feature[i] = 1
        X.append(feature)
    for i in pos_lines:
        obj = json.loads(i)
        feature_vec = np.array(obj['feature'])
        features = [feature_vec[i] for i in
                    ls_index]  # using all useful features from decision tree or single feature list skip this line
        feature = np.asarray(features)  # feature_vec instead of features
        feature = np.nan_to_num(feature.astype(np.float32))
        for i in range(len(feature)):
            if feature[i] > 205:
                feature[i] = 1
        X.append(feature)
    arr_x = np.array(X)
    arr_y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.2, shuffle=True, stratify=arr_y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def nn(X_train, X_test, y_train, y_test):
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model = Sequential()
    model.add(Dense(100, input_dim=len(X_train[0, :]), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(80, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(40, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(4, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=False, verbose=1)
    weights = model.get_weights()
    print(weights)
    y_pred = model.predict(X_test)
    y_pred_bool = []
    for i, predicted in enumerate(y_pred):
        if predicted[0] > 0.5:
            y_pred_bool.append(1)
        else:
            y_pred_bool.append(0)

    print(classification_report(y_test, y_pred_bool))
    loss, f1 = model.evaluate(X_test, y_test, verbose=0)
    return f1


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    X_train, X_test, y_train, y_test = prepare_data(neg_path, pos_path)
    f1 = nn(X_train, X_test, y_train, y_test)
    print(f1)

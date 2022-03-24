import json

import numpy as np
import yaml
from pkg_resources import resource_filename
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from subwords.visualation import get_pacmap_pca_tsne_word_vs_x

def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def prepare_data(neg_path, pos_path):
    feature_names = config['feature_names']
    print(feature_names)

    ls_index = []
    for i in feature_names:  # using different features
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
        feature = np.asarray(feature_vec)  # feature_vec instead of features
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
        feature = np.asarray(feature_vec)  # feature_vec instead of features
        feature = np.nan_to_num(feature.astype(np.float32))
        for i in range(len(feature)):
            if feature[i] > 205:
                feature[i] = 1
        X.append(feature)
    arr_x = np.array(X)
    arr_y = np.array(Y)

    # X_train, X_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.2, shuffle=True, stratify=arr_y)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    return arr_x


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    arr_x = prepare_data(neg_path, pos_path)
    print(len(arr_x))
    clustering = DBSCAN(eps=3, min_samples=5).fit(arr_x)
    labels = clustering.labels_
    print(len(labels))

    word_vec_list = []
    emb1 = []
    emb2 = []
    emb3 = []
    for index, label in enumerate(labels):
        point = arr_x[index]
        if label == -1:
            word_vec_list.append(point)
        elif label == 0 or 1 or 2 or 3 or 4:
            emb1.append(point)
        elif label == 5 or 6 or 7 or 8 or 9:
            emb2.append(point)
        else:
            emb3.append(point)
    other_emb = [emb1, emb2, emb3]
    print(len(other_emb))
    legend_names = ['Noise', '0-4', '5-9', '10-14']
    output_dir = resource_filename(__name__, config['output_dir']['path'])
    name_title = 'Cluster'
    get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)

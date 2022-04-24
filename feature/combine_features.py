import json

import numpy as np
import yaml
from pkg_resources import resource_filename


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    sen_embedding_path = resource_filename(__name__, config['sen_embedding_path']['path'])
    max_embedding_path = resource_filename(__name__, config['max_embedding_path']['path'])
    min_embedding_path = resource_filename(__name__, config['min_embedding_path']['path'])
    sum_embedding_path = resource_filename(__name__, config['sum_embedding_path']['path'])
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

    print(len(X))

    h = []
    with open(sen_embedding_path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        feature = np.array(obj['embedding'])
        feature = np.nan_to_num(feature.astype(np.float32))
        h.append(feature)
    print(len(h))

    a = []
    with open(max_embedding_path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        feature = np.array(obj['embedding'])
        feature = np.nan_to_num(feature.astype(np.float32))
        a.append(feature)
    print(len(a))

    b = []
    with open(min_embedding_path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        feature = np.array(obj['embedding'])
        feature = np.nan_to_num(feature.astype(np.float32))
        b.append(feature)
    print(len(b))

    c = []
    with open(min_embedding_path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        feature = np.array(obj['embedding'])
        feature = np.nan_to_num(feature.astype(np.float32))
        c.append(feature)
    print(len(c))
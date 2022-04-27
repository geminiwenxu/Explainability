import json

import numpy as np
import pandas as pd
import yaml
from pkg_resources import resource_filename


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def combine(path):
    ls = []
    text = []
    with open(path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        em_text = obj['text']
        em = np.array(obj['embedding'])
        em = np.nan_to_num(em.astype(np.float32))
        ls.append(em)
        text.append(em_text)
    df = pd.DataFrame(text, columns=['text'])
    df = df.join(pd.DataFrame(ls))
    return df


def wo_combine(path):
    ls = []
    with open(path) as f:
        embedding = f.readlines()
    for i in embedding:
        obj = json.loads(i)
        em = np.array(obj['embedding'])
        em = np.nan_to_num(em.astype(np.float32))
        ls.append(em)
    return pd.DataFrame(ls)


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    sen_embedding_path = resource_filename(__name__, config['sen_embedding_path']['path'])
    max_embedding_path = resource_filename(__name__, config['max_embedding_path']['path'])
    min_embedding_path = resource_filename(__name__, config['min_embedding_path']['path'])
    sum_embedding_path = resource_filename(__name__, config['sum_embedding_path']['path'])
    mean_embedding_path = resource_filename(__name__, config['mean_embedding_path']['path'])

    sen = combine(sen_embedding_path)
    max = wo_combine(max_embedding_path)
    min = wo_combine(min_embedding_path)
    sum = wo_combine(sum_embedding_path)
    mean = wo_combine(mean_embedding_path)
    df = pd.concat([sen, max, min, sum, mean], axis=1)
    df['em'] = df[df.columns[1:]].apply(lambda x: ','.join(x.astype(str)), axis=1)
    df = df[['text', 'em']]
    print(df)

    with open(neg_path) as n:
        feature_neg = n.readlines()
    with open(pos_path) as p:
        feature_pos = p.readlines()
    ls = []
    for idex, row in df.iterrows():
        em_text = row['text']
        em = (row['em'].split(','))
        em = [float(i) for i in em]
        print(em)
        print(type(em))
        for n in feature_neg:
            dict = {}
            neg_feature_obj = json.loads(n)
            neg_feature_text = neg_feature_obj['text']
            neg_feature = neg_feature_obj['feature']
            if em_text == neg_feature_text:
                print(em_text)
                dict['text'] = em_text
                dict['agg_em'] = em + neg_feature
                dict['sentiment'] = 0
                ls.append(dict)
        for p in feature_pos:
            dict = {}
            pos_feature_obj = json.loads(n)
            pos_feature_text = pos_feature_obj['text']
            pos_feature = pos_feature_obj['feature']
            if em_text == pos_feature_text:
                print(em_text)
                dict['text'] = em_text
                dict['agg_em'] = em + pos_feature
                dict['sentiment'] = 1
                ls.append(dict)

    output_file = open(resource_filename(__name__, config['agg_embeddings_file_path']['path']), 'w', encoding='utf-8')
    for dic in ls:
        json.dump(dic, output_file)
        output_file.write("\n")


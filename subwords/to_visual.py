import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from ml_classifier.svc import get_config
from subwords.visualation import get_pacmap_pca_tsne_word_vs_x


def extract_feature(path):
    df = pd.read_json(path, lines=True).sample(n=100, random_state=1)
    features = []
    for index, row in df.iterrows():
        cls = row['cls']
        max = row['max']
        min = row['min']
        avg = row['avg']
        sent = row['sent']
        maxim_feature = row['maxim_feature']
        bare_token_feature = row['bare_token_feature']
        ls = []
        for key in ['spacy_token', 'bert_tokens', 'bert_sub_tokens', 'PRON', 'AUX', 'ADJ', 'NOUN', 'PART', 'ADV', 'ADP',
                    'DET', 'PUNCT', 'VERB', 'CCONJ', 'NUM', 'X', 'SCONJ']:
            if key in bare_token_feature:
                ls.append(bare_token_feature[key])
            else:
                ls.append(0)
        feature = cls + max + min + avg + sent + maxim_feature + ls
        feature_arr = np.asarray(feature)
        feature_arr = np.nan_to_num(feature_arr.astype(np.float))
        for i in range(len(feature_arr)):
            if feature_arr[i] > 200:
                feature_arr[i] = 10
        feature = feature_arr.tolist()
        features.append(feature)
    features_arr = np.asarray(features)
    features_arr = np.nan_to_num(features_arr.astype(np.float))

    return features_arr


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    bert_feat_neg = resource_filename(__name__, config['bert_feat_neg']['path'])
    bert_feat_pos = resource_filename(__name__, config['bert_feat_pos']['path'])

    word_vec_list = extract_feature(bert_feat_neg).tolist()
    other_emb = [extract_feature(bert_feat_pos).tolist()]

    legend_names = ['neg', 'pos']
    output_dir = resource_filename(__name__, config['output_dir']['path'])
    name_title = 'Embeddings Cluster'
    get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)

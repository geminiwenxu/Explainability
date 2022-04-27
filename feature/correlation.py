import numpy as np
import pandas as pd
import yaml
from pkg_resources import resource_filename
from sklearn.cluster import DBSCAN

from subwords.visualation import get_pacmap_pca_tsne_word_vs_x

if __name__ == "__main__":
    def get_config(path: str) -> dict:
        with open(resource_filename(__name__, path), 'r') as stream:
            conf = yaml.safe_load(stream)
        return conf


    config = get_config('/../config/config.yaml')
    agg_embedding_path = resource_filename(__name__, config['agg_embeddings_file_path']['path'])
    df = pd.read_json(agg_embedding_path, lines=True)

    for index, row in df.iterrows():
        agg_em = row['agg_em']
    df_expend = df['agg_em'].apply(pd.Series)
    df = pd.concat([df['text'], df_expend, df['sentiment']], axis=1)


    df_array = df_expend.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    df_array = df_array.to_numpy()
    clustering = DBSCAN(eps=3, min_samples=5).fit(df_array)
    labels = clustering.labels_
    print('set of labels: ', set(labels))
    word_vec_list = []
    emb1 = []
    emb2 = []
    emb3 = []
    emb4 = []
    for index, label in enumerate(labels):
        point = df_array[index]
        if label == -1:
            word_vec_list.append(point)
        elif label == 0:
            emb1.append(point)
        elif label == 1:
            emb2.append(point)
        elif label == 2:
            emb3.append(point)
        else:
            emb4.append(point)
    other_emb = [emb1, emb2, emb3]

    # word_vec_list = neg_ls_sen
    # other_emb = [pos_ls_sen]
    legend_names = ['-1', '0', '1', '2', '3']
    output_dir = resource_filename(__name__, config['output_dir']['path'])
    name_title = 'Embeddings Cluster'
    get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)

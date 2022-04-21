import json

import pandas as pd
from pkg_resources import resource_filename
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertTokenizer

from ml_classifier.svc import get_config
from subwords.embedding import embeddings

bert_model = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def em_generator():
    em = []
    config = get_config('/../config/config.yaml')
    feature_neg_test_file_path = resource_filename(__name__, config['feature_neg_test_file_path']['path'])
    feature_pos_test_file_path = resource_filename(__name__, config['feature_pos_test_file_path']['path'])
    df = pd.read_csv(feature_neg_test_file_path, sep=';')
    neg_df = df.drop("Unnamed: 0", axis=1)
    pos_df = pd.read_csv(feature_pos_test_file_path, sep=',')

    with tqdm(total=neg_df.shape[0]) as pbar:
        for index, row in neg_df.iterrows():
            dict = {}
            pbar.update(1)
            neg_text_input = row['test_input']
            tokens = bert_model.tokenize(neg_text_input)
            if len(tokens) < 510:
                try:
                    dict['text'] = neg_text_input
                    # dict['embedding'] = model.encode(neg_text_input).tolist()
                    dict['embedding'] = embeddings(neg_text_input).tolist()
                    em.append(dict)
                except ValueError:
                    print('value error')

    with tqdm(total=pos_df.shape[0]) as pbar:
        for index, row in pos_df.iterrows():
            dict = {}
            pbar.update(1)
            pos_text_input = row['text']
            tokens = bert_model.tokenize(pos_text_input)
            if len(tokens) < 510:
                try:
                    dict['text'] = pos_text_input
                    # dict['embedding'] = model.encode(pos_text_input).tolist()
                    dict['embedding'] = embeddings(pos_text_input).tolist()
                    em.append(dict)
                except ValueError:
                    print('value error')

    output_file = open(resource_filename(__name__, config['embedings_file_path']['path']), 'w', encoding='utf-8')
    for dic in em:
        json.dump(dic, output_file)
        output_file.write("\n")

    # clustering = DBSCAN(eps=3, min_samples=5).fit(ls_sen)
    # labels = clustering.labels_
    # print('set of labels: ', set(labels))
    # word_vec_list = []
    # emb1 = []
    # emb2 = []
    # emb3 = []
    # for index, label in enumerate(labels):
    #     point = ls_sen[index]
    #     if label == -1:
    #         word_vec_list.append(point)
    #     elif label == 0:
    #         emb1.append(point)
    #     elif label == 1:
    #         emb2.append(point)
    #     else:
    #         emb3.append(point)
    # other_emb = [emb1, emb2, emb3]
    # print(len(other_emb))
    # word_vec_list = neg_ls_sen
    # other_emb = [pos_ls_sen]
    # legend_names = ['neg', 'pos']
    # output_dir = resource_filename(__name__, config['output_dir']['path'])
    # name_title = 'Embeddings Cluster'
    # get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)


if __name__ == "__main__":
    em_generator()

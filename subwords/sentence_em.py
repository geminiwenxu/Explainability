from ml_classifier.svc import get_config
import pandas as pd
from pkg_resources import resource_filename
from subwords.embedding import embeddings
from transformers import BertTokenizer
import re
from subwords.visualation import get_pacmap_pca_tsne_word_vs_x
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import spacy


def sentence_em():
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_file_path']['path'])
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])
    feature_test_file_path = resource_filename(__name__, config['feature_test_file_path']['path'])
    german_stop_words = stopwords.words('german')

    df = pd.read_csv(feature_test_file_path, sep=';')
    result = df.drop("Unnamed: 0", axis=1)
    ls_wrong = []
    ls_correct = []
    for index, row in result.iterrows():
        neg_text_input = row['test_input']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(neg_text_input)
        # print(neg_text_input)
        sen_split = re.findall(r'\w+|\S+', neg_text_input)
        low_sen_split = [i.lower() for i in sen_split]
        # low_sen_split.remove('.')
        # print(low_sen_split)
        if len(tokens) < 510:
            try:
                sentence_embeddings, tokenized_text = embeddings(neg_text_input)
                # tokenized_text.remove('.')
                common_tokens = list(set(tokenized_text).intersection(low_sen_split))
                diff_tokens = list(set(tokenized_text) - set(low_sen_split))
                # print(sentence_embeddings)
                # print(tokenized_text)
                # print("common tokens: ", common_tokens)
                # print("different tokens: ", diff_tokens)
                # print(sentence_embeddings)
                # print(tokenized_text)
                # print(common_tokens)
                # print(diff_tokens)
                wrong_token = []
                for i in diff_tokens:
                    index = tokenized_text.index(i)
                    token_em = sentence_embeddings[index]
                    wrong_token.append(token_em)

                correct_token = []
                for i in common_tokens:
                    index = tokenized_text.index(i)
                    token_em = sentence_embeddings[index]
                    correct_token.append(token_em)

                wrong = []
                for i in range(0, len(wrong_token)):
                    arr = wrong_token[i].numpy()
                    wrong.append(arr)
                correct = []
                for i in range(0, len(correct_token)):
                    arr = correct_token[i].numpy()
                    correct.append(arr)
            except ValueError:
                print('value error')
        ls_wrong.append(wrong)
        ls_correct.append(correct)
    flat_ls_wrong = [item for sublist in ls_wrong for item in sublist]
    flat_ls_correct = [item for sublist in ls_correct for item in sublist]
    print("flag", flat_ls_wrong)
    print("correct", flat_ls_correct)
    legend_names = ['wrong', 'correct']
    output_dir = resource_filename(__name__, config['output_dir']['path'])
    name_title = 'subword embeddings'
    get_pacmap_pca_tsne_word_vs_x(flat_ls_wrong, [flat_ls_correct], legend_names, output_dir, name_title)

    # df = pd.read_csv(feature_pos_file_path, sep=',')
    # for index, row in df.iterrows():
    #     pos_text_input = row['text']
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     tokens = tokenizer.tokenize(pos_text_input)
    #     if len(tokens) < 510:
    #         try:
    #             embeddings(pos_text_input)
    #         except ValueError:
    #             print('value error')


if __name__ == "__main__":
    sentence_em()

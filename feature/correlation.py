# correlations = cdist(df.iloc[:, :-1].to_numpy().transpose(), df['Sentiment'].to_numpy().reshape(1, -1),
#                      metric='correlation')
# print(correlations)

# df_array = df.to_numpy()
# clustering = DBSCAN(eps=3, min_samples=5).fit(df_array)
# labels = clustering.labels_
# print('set of labels: ', set(labels))
# word_vec_list = []
# emb1 = []
# emb2 = []
# emb3 = []
# for index, label in enumerate(labels):
#     point = df_array[index]
#     if label == -1:
#         word_vec_list.append(point)
#     elif label == 0:
#         emb1.append(point)
#     elif label == 1:
#         emb2.append(point)
#     else:
#         emb3.append(point)
# other_emb = [emb1, emb2, emb3]
#
# # word_vec_list = neg_ls_sen
# # other_emb = [pos_ls_sen]
# legend_names = ['0', '1', '2', '-1']
# output_dir = resource_filename(__name__, config['output_dir']['path'])
# name_title = 'Embeddings Cluster'
# get_pacmap_pca_tsne_word_vs_x(word_vec_list, other_emb, legend_names, output_dir, name_title)
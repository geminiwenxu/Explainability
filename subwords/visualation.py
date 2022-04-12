from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pacmap
import os
import pandas as pd
import numpy as np


def get_pacmap_pca_tsne_word_vs_x(word_vec_list: list, other_emb: list, legend_names: list, output_dir: str,
                                  name_title: str):
    y_list = []
    x_list = []
    label = 0
    # vectors for words list[vec1, vec2, vec3, ...]
    for word_vec_i in word_vec_list:
        y_list.append(label)
        x_list.append(word_vec_i)
    # if you have other embeddings emb[emb1[vec1, vec2,...], emb2[vec1, vec2], ...]
    for emb_i in other_emb:
        label += 1
        for vec_i in emb_i:
            y_list.append(label)
            x_list.append(vec_i)
    pca_transformer = PCA(n_components=2)
    pac_map = pacmap.PaCMAP(n_dims=2, n_neighbors=None)
    t_sne_transformer = TSNE(n_components=2, n_jobs=6)
    X_r_pca = pca_transformer.fit_transform(x_list)
    X_r_t_sne = t_sne_transformer.fit_transform(x_list)
    X_r_pacmap = pac_map.fit_transform(pd.DataFrame(x_list).values)
    out_pca = output_dir.replace(".png", "_PCA.png")
    out_pacmap = output_dir.replace(".png", "_PaCMAP.png")
    out_tsne = output_dir.replace(".png", "_TSNE.png")
    os.makedirs(os.path.dirname(out_pca), exist_ok=True)
    get_visualisation(X_r_pca, y_list, legend_names, out_pca, f"PCA {name_title}")
    # get_visualisation(X_r_pacmap, y_list, legend_names, out_pacmap, f"PaCMAP {name_title}")
    # get_visualisation(X_r_t_sne, y_list, legend_names, out_tsne, f"TSNE {name_title}")


def get_visualisation(X_r, labels, legend_names, output_dir, name_title):
    color_labels = ["red", "green", "blue", "saddlebrown", "indigo", "gray", "darkorange", "gold", "olive",
                    "aquamarine", "steelblue", "blueviolet", "rosybrown"]
    # X_r = transformers.fit_transform(embeddings)
    lw = 2
    fig, ax = plt.subplots()
    x_axis = []
    y_axis = []
    for i in range(0, len(legend_names)):
        y_axis.append([])
        x_axis.append([])
    for c in range(0, len(X_r)):
        x_axis[labels[c]].append(X_r[c][0])
        y_axis[labels[c]].append(X_r[c][1])
    for index, target_name in zip(range(0, len(x_axis)), legend_names):
        ax.scatter(x_axis[index], y_axis[index], c=color_labels[index], alpha=0.5, lw=lw,
                   label=target_name,
                   s=10)
    legend_ax = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, shadow=True, scatterpoints=1)
    title_ax = ax.set_title(f"{name_title}")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # plt.show()
    plt.savefig(output_dir, dpi=fig.dpi, bbox_extra_artists=(legend_ax, title_ax), bbox_inches='tight')
    plt.cla()
    plt.close(fig)





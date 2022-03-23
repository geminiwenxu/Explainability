from sklearn.cluster import DBSCAN
from subwords.sentence_em import sentence_em

if __name__ == "__main__":
    flat_ls_wrong, flat_ls_correct = sentence_em()
    clustering = DBSCAN(eps=3, min_samples=5).fit(flat_ls_wrong)
    print(clustering.labels_)

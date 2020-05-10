import numpy as np
from ours_eval import csls_knn_10_score, evaluation, separate_eva
from icp import ICPTrainer
import matplotlib.pyplot as plt
import argparse
import utils
from utils import sub_icp, find_clts, find_centers, csls_knn_10_score
import params
import sklearn.cluster
from sklearn.decomposition import PCA


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', type=int, default=2, help='The number of cluster to be divided')
    args = parser.parse_args()
    n_clusters = args.cluster_num

    src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_eval_ex, False) #n_eval_ex = 200000
    tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_eval_ex, False)
    src_normed = normalize(src_embeddings)
    tgt_normed = normalize(tgt_embeddings)
    cross_dict_src2tgt = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang), src_word2id, tgt_word2id)
    cross_dict_tgt2src = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.tgt_lang, params.src_lang), tgt_word2id, src_word2id)

    T = np.load("%s/%s_%s_T.npy" % (params.cp_dir, params.src_lang, params.tgt_lang))
    T2 = np.load("%s/%s_%s_T.npy" % (params.cp_dir, params.tgt_lang, params.src_lang))
    TranslatedX = src_embeddings.dot(np.transpose(T))

    src_full = np.load("data/%s_%d.npy" % (params.src_lang, params.n_init_ex)) # 10000, 10000 english
    src_trans = src_full.dot(np.transpose(T))
    tgt_full = np.load("data/%s_%d.npy" % (params.tgt_lang, params.n_init_ex)) # 300, 10000 es

    src_trans_normed = normalize(src_trans)
    tgt_full_normed = normalize(tgt_full)

    src_clt = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init= 40, random_state=200)
    src_y = src_clt.fit_predict(src_trans_normed)
    tgt_y = src_clt.predict(tgt_full_normed)

    src_centers, tgt_centers, src_dic, tgt_dic = find_centers(src_full, tgt_full, src_y, tgt_y, n_clts = n_clusters)
    Translated_centers = src_centers.dot(np.transpose(T))

    trans_c = normalize(Translated_centers)
    tgt_c = normalize(tgt_centers)

    cross_dict_src2tgt = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang), src_word2id, tgt_word2id)
    cross_dict_tgt2src = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.tgt_lang, params.src_lang), tgt_word2id, src_word2id)

    src_classes = find_clts(data = src_embeddings, centers = src_centers, dico = cross_dict_src2tgt)
    tgt_classes = find_clts(data = tgt_embeddings, centers = tgt_centers, dico = cross_dict_tgt2src)

    src_classes_trans = find_clts(data = src_embeddings, centers = src_centers, dico = cross_dict_src2tgt, trans = True)
    src_correct = np.where(src_classes == src_classes_trans)
    src_acc = src_correct[0].shape[0]/1500
    tgt_classes_trans = find_clts(data = tgt_embeddings, centers = tgt_centers, dico = cross_dict_tgt2src, trans = True)
    tgt_correct = np.where(tgt_classes == tgt_classes_trans)
    tgt_acc = tgt_correct[0].shape[0]/1500
    print(src_acc, tgt_acc)

    TX, TY = utils.multi_ICP(src_full, tgt_full, src_y, tgt_y, src_dic, tgt_dic, n_clusters, time_run_icp = 100)

    result_src = separate_eva(src_embeddings, tgt_embeddings, T, TX, src_classes, dico=cross_dict_src2tgt)
    result_tgt = separate_eva(tgt_embeddings, src_embeddings, T2, TY, tgt_classes, dico=cross_dict_tgt2src)


if __name__ == '__main__':
    main()   

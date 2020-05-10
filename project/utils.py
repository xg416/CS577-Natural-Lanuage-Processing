# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import params
from icp import ICPTrainer
import os
import io
import numpy as np
import faiss
import operator
import time
import multiprocessing

def read_txt_embeddings(emb_path, max_vocab, full_vocab=False):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in embedding file" &(word))
                else:
                    if not vect.shape == (300,):
                        print("Invalid dimension (%i) for word '%s' in line %i."
                                       % (vect.shape[0], word, i))
                        continue
                    assert vect.shape == (300,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    assert embeddings.shape == (len(id2word), 300)
    return id2word, word2id, embeddings


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    print(path)
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append(np.array([word2id1[word1], word2id2[word2]]))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    print("Found %i pairs of words in the dictionary (%i unique). "
          "%i other pairs contained at least one unknown word "
          "(%i in lang1, %i in lang2)"
          % (len(pairs), len(set([x for x, _ in pairs])),
             not_found, not_found1, not_found2))

    pairs = np.array(pairs)
    return pairs


# def get_nn_avg_dist(emb, query, knn):
#     # cpu mode
#     index = faiss.IndexFlatIP(emb.shape[1])
#     index.add(emb)
#     distances, _ = index.search(query, knn)
#     return distances.mean(1)
def get_nn_avg_dist(emb, query, knn):
    # cpu mode
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.IndexFlatIP(emb.shape[1])
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb)
    distances, _ = index.search(query, knn)
    return distances.mean(1)

def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """

    assert np.max(dico[:, 0]) < len(emb1)
    assert np.max(dico[:, 1]) < len(emb2)

    # normalize word embeddings
    emb1 = emb1 / np.linalg.norm(emb1, ord=2, axis=1, keepdims=True)
    emb2 = emb2 / np.linalg.norm(emb2, ord=2, axis=1, keepdims=True)
    emb1 = emb1.astype('float32')
    emb2 = emb2.astype('float32')

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.dot(emb2.T)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn) #(200000,)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn) #(200000,)
        # queries / scores
        query = emb1[dico[:, 0]] #dico:2975,2. it is a mapping index pair. 
        scores = 2 * query.dot(emb2.T) #2975*200000
        scores -= average_dist1[dico[:, 0]][:, None] # right hand side: 2975, 1
        scores -= average_dist2[None, :] # right hand side: 1, 200000

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = np.argsort(-scores, 1)[:, :10] #2975x10
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None]).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("%i source words - %s - Precision at k = %i: %f" %
              (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results


def sub_icp(src_W, tgt_W, n_icp_runs):
    def run_icp(s0, i):
        np.random.seed(s0 + i)
        icp = ICPTrainer(src_W.copy(), tgt_W.copy(), True, params.n_pca)
        t0 = time.time()
        # np.random.seed(50007)
        indices_x, indices_y, rec, bb = icp.train_icp(params.icp_init_epochs)
        dt = time.time() - t0
        print("%d: Rec %f BB %d Time: %f" % (i, rec, bb, dt))
        return indices_x, indices_y, rec, bb
    data = np.zeros((n_icp_runs, 2)) #100, 2

    best_idx_x = None
    best_idx_y = None

    min_rec = 1e8
    s0 = np.random.randint(50000)
    results = []
    if params.n_processes == 1:
        for i in range(n_icp_runs):
            results += [run_icp(s0, i)]
    else:
        pool = multiprocessing.Pool(processes=params.n_processes)
        for result in tqdm.tqdm(pool.imap_unordered(run_icp, range(n_icp_runs)), total=n_icp_runs):
            results += [result]
        pool.close()

    min_rec = 1e8
    min_bb = None
    for i, result in enumerate(results):
        indices_x, indices_y, rec, bb = result
        data[i, 0] = rec
        data[i, 1] = bb
        if rec < min_rec:
            best_idx_x = indices_x
            best_idx_y = indices_y
            min_rec = rec
            min_bb = bb


    idx = np.argmin(data[:, 0], 0)
    print("Init - Achieved: Rec %f BB %d" % (data[idx, 0], data[idx, 1]))
    icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
    _, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, True, best_idx_x, best_idx_y)
    print("Training - Achieved: Rec %f BB %d" % (rec, bb))
    icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
    icp_ft.icp.TX = icp_train.icp.TX
    icp_ft.icp.TY = icp_train.icp.TY
    _, _, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)

    print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
    TX = icp_ft.icp.TX
    TY = icp_ft.icp.TY
    return TX, TY


def find_clts(data, centers, dico, trans = False):
    if trans:
        samples = data[dico[:,1],:]
    else:
        samples = data[dico[:,0],:]
    samples = np.repeat(samples[:, np.newaxis, :], centers.shape[0], axis = 1)
    diff = samples - centers
    dist = np.linalg.norm(diff, ord=2, axis=2)
    classes = dist.argmin(axis = 1)
    return classes


def find_centers(src, tgt, src_y, tgt_y, n_clts):
    dic = {}
    for i in src_y:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
    src_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse = True)
    print(src_dic)
    src_centers = []
    for i in range(n_clts):
        idx = src_y == src_dic[i][0]
        src_centers.append(np.mean(src[idx], axis = 0))

    dic = {}
    for i in tgt_y:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
    tgt_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse = True)
    print(tgt_dic)
    tgt_centers = []
    for i in range(n_clts):
        idx = tgt_y == tgt_dic[i][0]
        tgt_centers.append(np.mean(tgt[idx], axis = 0))
    src_centers = np.asarray(src_centers)
    tgt_centers = np.asarray(tgt_centers)
    return src_centers, tgt_centers, src_dic, tgt_dic


def csls_knn_10_score(emb_trans, emb_tgt, dico):
    emb_trans = emb_trans / np.linalg.norm(emb_trans, ord=2, axis=1, keepdims=True)
    emb_tgt = emb_tgt / np.linalg.norm(emb_tgt, ord=2, axis=1, keepdims=True)
    emb_trans = emb_trans.astype('float32')
    emb_tgt = emb_tgt.astype('float32')
    # I use csls_knn_10 directly
    average_dist1 = get_nn_avg_dist(emb = emb_tgt, query = emb_trans, knn = 10) #(200000,)
    average_dist2 = get_nn_avg_dist(emb = emb_trans, query = emb_tgt, knn = 10) #(200000,)
    
    query = emb_trans[dico[:, 0]] # dico[:, 0] is from source Domain, # dico[:, 1] is from target domain
    scores = 2 * query.dot(emb_tgt.T) #2975*200000
    scores -= average_dist1[dico[:, 0]][:, None] # right hand side: 2975, 1
    scores -= average_dist2[None,:] # right hand side: 1, 200000
    
    return scores

def multi_ICP(src_full, tgt_full, src_y, tgt_y, src_y_dic, tgt_y_dic, n_clusters, time_run_icp):
    TX = []
    TY = []
    for i in range(n_clusters):
        src_idx = (src_y == src_y_dic[i][0])
        src = src_full[src_idx]
        tgt_idx = (tgt_y == tgt_y_dic[i][0])
        tgt = tgt_full[tgt_idx]
        print("cluster", i, "src shape:", src.shape, "tgt shape:", tgt.shape)
        X, Y = sub_icp(src.T, tgt.T, time_run_icp)
        TX.append(X)
        TY.append(Y)
    return TX, TY

def translate(src_embeddings, global_T, TX, classes, dico):
    Translated = src_embeddings.dot(np.transpose(global_T))
    for i in range(len(TX)):
        # class_i = classes[classes==i]
        src_idx = np.where(classes==i)
        translated_idx = dico[src_idx, 0]
        Translated_i = src_embeddings.dot(np.transpose(TX[i]))
        Translated[translated_idx] = Translated_i[translated_idx]
    return Translated
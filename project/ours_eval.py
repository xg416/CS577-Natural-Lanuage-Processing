import numpy as np
import utils
import params
from utils import get_nn_avg_dist

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
    
def evaluation_save(scores, dico, tgt_emb):
    results = []
    top_matches = np.argsort(-scores, 1)[:, :10] #2975x10
    np.save("top_matches", top_matches)
    np.save("gt",dico[:, 1][:, None])
    for k in [1, 5]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None]).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):# dico[:, 0] has only 1500 words
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("%i source words - %s - Precision at k = %i: %f" %
              (len(matching), " ", k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results    
    
def evaluation(scores, dico):
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
              (len(matching), " ", k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results


def separate_eva(src_embeddings, emb_tgt, global_T, TX, classes, dico):
    #Translated = src_embeddings.dot(np.transpose(global_T))
    result = np.asarray([0,0,0], dtype= 'float64')
    len_dico = dico.shape[0]
    for i in range(len(TX)):
        # class_i = classes[classes==i]
        src_idx = np.where(classes==i)
        sub_dico = dico[src_idx]
        len_i = sub_dico.shape[0]
        # translated_idx = sub_dico[:, 0]
        Translated_i = src_embeddings.dot(np.transpose(TX[i]))
        #Translated[translated_idx] = Translated_i[translated_idx]
        scores_i = csls_knn_10_score(emb_trans=Translated_i, emb_tgt=emb_tgt, dico=sub_dico)
        print('result of the cluster', i)
        result_i = evaluation(scores_i, sub_dico)
        result += np.asarray([result_i[0][1]*len_i, result_i[1][1]*len_i, result_i[2][1]*len_i], dtype= 'float64')
    result /= len_dico
    print("the final result:")
    for i, k in enumerate([1, 5, 10]):
        print("%i source words - %s - Precision at k = %i: %f" %
            (len_dico, " ", k, result[i]))
    return result



if __name__ == "__main__":
    src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_eval_ex, False) #n_eval_ex = 200000
    tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_eval_ex, False)

    scores_src2tgt = np.load('data/scores-%s-%s.vec' % (params.src_lang, params.tgt_lang))
    scores_tgt2src = np.load('data/scores-%s-%s.vec' % (params.tgt_lang, params.src_lang))
    cross_dict_src2tgt = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang), src_word2id, tgt_word2id)
    cross_dict_tgt2src = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.tgt_lang, params.src_lang), tgt_word2id, src_word2id)

    evaluation(scores_src2tgt,cross_dict_src2tgt)
    evaluation(scores_tgt2src,cross_dict_tgt2src)


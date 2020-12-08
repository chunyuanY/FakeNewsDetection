import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import jieba
jieba.set_dictionary('dict.txt.big')
np.random.seed(0)

w2v_dim = 300

dic = {
    'non-rumor': 0,   # Non-rumor   NR
    'false': 1,   # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,    # debunk rumor  TR
}

def clean_str_cut(string, task):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'m", " am", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " have", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", " are", string)
        string = re.sub(r"\'d", " had", string)
        string = re.sub(r"\'ll", " will", string)

    # string = re.sub(r"\.", " . ", string)
    string = re.sub(r"'", " ' ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"#", " # ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower())) if task == "weibo" else string.strip().lower().split()
    return words


def construct_normalized_adj(R, shape, normalize=True, symmetry=False):
    adj = sp.csc_matrix((R[:, 2], (R[:, 0], R[:, 1])), shape=shape, dtype=np.float32)
    # adj = sp.coo_matrix(R, dtype=np.float32)
    def normalize_adj(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    if normalize and not symmetry:
        if adj.shape[0] == adj.shape[1]:
            adj = adj + sp.eye(adj.shape[0])
        # Row-normalize sparse matrix
        rowsum = np.array(adj.sum(1))
        D_row = np.power(rowsum, -0.5).flatten()
        D_row[np.isinf(D_row)] = 0.
        D_row = sp.diags(D_row)

        colsum = np.array(adj.sum(0))
        D_col = np.power(colsum, -0.5).flatten()
        D_col[np.isinf(D_col)] = 0.
        D_col = sp.diags(D_col)
        adj = adj.dot(D_col).transpose().dot(D_row).transpose() # .tocsr()
    elif normalize and symmetry:
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return adj


def read_text_data(X_sids, root_path, file_name, apstr):
    with open(root_path + file_name +apstr, 'r', encoding='utf-8') as input:
        X_uid, X_sid, X_source_wid, y_, y_user = [], [], [], [], []
        for line in input.readlines():
            uid, sid, content, label = line.strip().split("\t")
            X_sids.append(sid)
            X_uid.append(str(uid))
            X_sid.append(sid)
            X_source_wid.append(clean_str_cut(content, file_name))
            y_.append(dic[label])
    return X_uid, X_sid, X_source_wid, y_


def read_structural_data(X_sids, X_train_sid, X_dev_sid, root_path):
    US_relation = []  # users post tweet
    UU_relation = []
    X_train_ruid, X_dev_ruid, X_test_ruid = [], [], []
    for file_name in X_sids:
        with open(root_path + "/tree/"+ file_name +".txt", 'r', encoding='utf-8') as input:
            while True:
                line0 = input.readline()
                if line0.__contains__("ROOT"): break

            uid0, sid0, _ = eval(line0.strip().split("->")[1])  # ['ROOT', 'ROOT', '0.0']->['972651', '80080680482123777', '0.0']
            US_relation.append([str(uid0), sid0])  # (uid, sid)

            X_ruid_i = []

            for line in input.readlines():
                arr = line.strip().split("->")
                uid1, sid1, t1 = eval(arr[0])
                uid2, rid2, t2 = eval(arr[1])

                X_ruid_i.append(str(uid2))
                UU_relation.append([str(uid1), str(uid2)])


            if file_name in X_train_sid:
                X_train_ruid.append(X_ruid_i)
            elif file_name in X_dev_sid:
                X_dev_ruid.append(X_ruid_i)
            else:
                X_test_ruid.append(X_ruid_i)

    return X_train_ruid, X_dev_ruid, X_test_ruid, US_relation, UU_relation # All_UU_relations  # Sids,


def read_corpus(root_path, file_name):
    X_sids = []

    X_train_uid, X_train_sid, X_train_source_wid, y_train = read_text_data(X_sids, root_path, file_name, ".train")
    X_dev_uid, X_dev_sid, X_dev_source_wid, y_dev = read_text_data(X_sids, root_path, file_name, ".dev")
    X_test_uid, X_test_sid, X_test_source_wid, y_test = read_text_data(X_sids, root_path, file_name, ".test")
    X_train_ruid, X_dev_ruid, X_test_ruid, US_relation, UU_relation = read_structural_data(X_sids, X_train_sid, X_dev_sid, root_path)

    sid_counter = Counter(X_sids)
    X_sids = [k for k, v in sid_counter.items()]
    Sids = {idx:i for i, idx in enumerate(X_sids)}
    pickle.dump(Sids, open(file_name+"/Sids.pkl", 'wb'))

    uid_counter1 = Counter(itertools.chain(*(X_train_ruid + X_dev_ruid + X_test_ruid)))
    X_ruids = [k for k, v in uid_counter1.most_common() if v >= 10]
    print("X_ruids: ", len(X_ruids))

    uid_counter2 = Counter(X_ruids + X_train_uid + X_dev_uid + X_test_uid)
    X_uids = [k for k, v in uid_counter2.most_common()]

    Uids = {idx:i+1 for i, idx in enumerate(X_uids)}
    pickle.dump(Uids, open(file_name+"/Uids.pkl", 'wb'))

    X_train_uid = [Uids[uid] for uid in X_train_uid]
    X_dev_uid = [Uids[uid] for uid in X_dev_uid]
    X_test_uid = [Uids[uid] for uid in X_test_uid]

    max_ret = 20
    X_train_ruid = [[Uids[uid] for uid in uids[:max_ret] if uid in Uids] for uids in X_train_ruid]
    X_dev_ruid = [[Uids[uid] for uid in uids[:max_ret] if uid in Uids] for uids in X_dev_ruid]
    X_test_ruid = [[Uids[uid] for uid in uids[:max_ret] if uid in Uids] for uids in X_test_ruid]

    X_train_ruid = [uids+[0]*(max_ret-len(uids)) for uids in X_train_ruid]
    X_dev_ruid   = [uids+[0]*(max_ret-len(uids)) for uids in X_dev_ruid]
    X_test_ruid  = [uids+[0]*(max_ret-len(uids)) for uids in X_test_ruid]

    US_relation = np.array([(Uids[u], Sids[s], 1) for u, s in US_relation if u in Uids and s in Sids])
    A_us = construct_normalized_adj(US_relation, shape=(len(Uids)+1, len(Sids)))
    print(A_us.shape)

    uu_counter = Counter([(Uids[u1], Uids[u2]) for u1, u2 in UU_relation if u1 in Uids and u2 in Uids])
    UU_relation = np.array([list(k) + [v] for k, v in uu_counter.items()])
    A_uu = construct_normalized_adj(UU_relation, shape=(len(Uids)+1, len(Uids)+1))
    print(A_uu.shape)

    with open(root_path + "user_credibility.txt", 'r', encoding='utf8') as fin:
        line = " ".join(fin.readlines())
        user_credibility = {Uids[k]: v for k, v in dict(eval(line)).items() if k in Uids}
        y_train_cred = []
        for uid in X_train_uid:
            y_train_cred.append(user_credibility[uid])

        y_train_rucred = []
        for uids in X_train_ruid:
            rucred = []
            for uid in uids:
                if uid == 0:
                    rucred.append(3)
                else:
                    rucred.append(user_credibility[uid])
            y_train_rucred.append(rucred)

    # A_uus = []
    # for UU_relation in All_UU_relations:
    #     uu_counter = Counter([(Uids[u1], Uids[u2]) for u1, u2 in UU_relation if u1 in Uids and u2 in Uids])
    #     UU_relation = np.array([list(k) + [v] for k, v in uu_counter.items()])
    #     A_uu = construct_normalized_adj(UU_relation, shape=(len(Uids)+1, len(Uids)+1), symmetry=True)
    #     A_uus.append(A_uu)

    X_train_sid = np.array([Sids[sid] for sid in X_train_sid])
    X_dev_sid = np.array([Sids[sid] for sid in X_dev_sid])
    X_test_sid = np.array([Sids[sid] for sid in X_test_sid])

    return X_train_source_wid, X_train_sid, X_train_uid, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
           X_dev_source_wid, X_dev_sid, X_dev_uid, X_dev_ruid, y_dev, \
           X_test_source_wid, X_test_sid, X_test_uid, X_test_ruid, y_test, \
           A_us, A_uu


def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            #add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)
            # print(word)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 2]  #
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)     #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return vocabulary, embedding_weights


def pad_sequence(X, max_len=50):
    X_pad = []
    for doc in X:
        if len(doc) >= max_len:
            doc = doc[:max_len]
        else:
            doc = [0] * (max_len - len(doc)) + doc
        X_pad.append(doc)
    return X_pad


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
    x = pad_sequence(x)
    return x


def data_extract(root_path, filename, w2v_path):
    X_train_source_wid, X_train_sid, X_train_uid, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    X_dev_source_wid, X_dev_sid, X_dev_uid, X_dev_ruid, y_dev, \
    X_test_source_wid, X_test_sid, X_test_uid, X_test_ruid, y_test, \
    A_us, A_uus = read_corpus(root_path, filename)

    print("text word2vec generation.......")
    vocabulary, word_embeddings = build_vocab_word2vec(X_train_source_wid + X_dev_source_wid + X_test_source_wid, w2v_path=w2v_path)
    pickle.dump(vocabulary, open(root_path + "/vocab.pkl", 'wb'))
    print("Vocabulary size: "+str(len(vocabulary)))

    print("build input data.......")
    X_train_source_wid = build_input_data(X_train_source_wid, vocabulary)
    X_dev_source_wid = build_input_data(X_dev_source_wid, vocabulary)
    X_test_source_wid = build_input_data(X_test_source_wid, vocabulary)

    pickle.dump([A_us, A_uus], open(root_path+"/relations.pkl", 'wb') )
    pickle.dump([X_train_source_wid, X_train_sid, X_train_uid, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings], open(root_path+"/train.pkl", 'wb') )
    pickle.dump([X_dev_source_wid, X_dev_sid, X_dev_uid, X_dev_ruid, y_dev], open(root_path+"/dev.pkl", 'wb') )
    pickle.dump([X_test_source_wid, X_test_sid, X_test_uid, X_test_ruid, y_test], open(root_path+"/test.pkl", 'wb') )



if __name__ == "__main__":
    # data_extract('./twitter15/', "twitter15", "twitter_w2v.bin")
    # data_extract('./twitter16/', "twitter16", "twitter_w2v.bin")
    data_extract('./weibo/', "weibo", "weibo_w2v.bin")




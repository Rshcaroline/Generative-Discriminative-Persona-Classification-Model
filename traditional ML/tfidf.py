import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
import pickle

train_path = './shran/data/bt_aug_clip_train4.tsv'
dev_path = './shran/data/bt_aug_clip_test4.tsv'
test_path = './shran/data/bt_aug_clip_dev4.tsv'

def get_freq_dict(f=train_path):
    f = open(f)
    freq_dict = {}
    speaker_freq_dict = {}
    for line in f.readlines():
        line = line.split('\t')
        response = line[-1].strip()
        words = response.split()
        for word in words:
            if word not in freq_dict:
                freq_dict[word] = 0
            freq_dict[word] += 1
        speaker = line[-2].strip()
        if speaker not in speaker_freq_dict:
            speaker_freq_dict[speaker] = 0
        speaker_freq_dict[speaker] += 1
    speaker_freq_dict = dict(sorted(speaker_freq_dict.items(),key=lambda x:-x[1])[:7])
    f.close()
    return freq_dict,speaker_freq_dict

def read_xy(file,s2idx):
    X, y = [], []
    for line in file.readlines():
        line = line.split('\t')
        spk = line[-2].strip()
        if spk not in s2idx:
            continue
        X.append(line[-1].strip())
        y.append(s2idx[spk])
    return X,y

def read_xy_k(file,s2idx,k):
    X, y = [],[]
    buffer = {}
    for i in range(len(s2idx)):
        buffer[i] = []
    for line in file.readlines():
        line = line.split('\t')
        spk = line[-2].strip()
        if spk not in s2idx:
            continue
        spk_idx = s2idx[spk]
        buffer[spk_idx].append(line[-1].strip())
        if len(buffer[spk_idx]) == k:
            X.append(' '.join(buffer[spk_idx]))
            y.append(spk_idx)
            buffer[spk_idx] = []
    return X,y

def tf_idf_clf(train=train_path, test=test_path,load=False,save_dir=''):
    freq_dict,speaker_freq_dict = get_freq_dict(train)
    word2idx, idx2word = {}, {}
    s2idx,idx2s = {},{}
    for i, w in enumerate(freq_dict):
        word2idx[w], idx2word[i] = i, w
    for i, w in enumerate(speaker_freq_dict):
        s2idx[w],idx2s[i] = i,w
    train,test = open(train),open(test)
    train_X,train_y = read_xy(train,s2idx)
    train_y = np.array(train_y)
    count_vec = CountVectorizer()
    train_X_cnt = count_vec.fit_transform(train_X)
    tf_trans = TfidfTransformer(use_idf=False).fit(train_X_cnt)
    train_X_tf = tf_trans.transform(train_X_cnt)
    # print(train_X_tf.shape)
    if not load:
        # clf = LogisticRegression(max_iter=10000).fit(train_X_tf,train_y)   # 0.257902768694
        # clf = MultinomialNB().fit(train_X_tf, train_y)   # 0.232831916285
        clf = RandomForestClassifier().fit(train_X_tf, train_y)   # 0.226073686505
        # clf = SVC().fit(train_X_tf, train_y)   # 0.232742090125
        if save_dir:
            pickle.dump(clf,open(save_dir,'wb'))
    else:
        clf = pickle.load(save_dir)
    test_X,test_y = read_xy_k(test,s2idx,k=1)
    test_y = np.array(test_y)
    test_X_cnt = count_vec.transform(test_X)
    test_X_tf = tf_trans.transform(test_X_cnt)

    predicted = clf.predict(test_X_tf)
    accur = accuracy_score(test_y,predicted)
    cm = confusion_matrix(test_y,predicted)
    print(accur)
    print(cm)

if __name__ == '__main__':
    tf_idf_clf(load=False, save_dir='./model/tfidf_rf.pkl')
# coding: utf-8

import jieba  # 处理中文
import matplotlib.pyplot as plt
import nltk  # 处理英文
import pandas as pd
import sklearn
from sklearn.naive_bayes import MultinomialNB

from config import *


# 停用词去重
def make_word_set(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


# 生成样本
def text_processing(pospath='./data/pos60000.txt',negpath='./data/neg60000.txt', test_size=0.2):

    data_list = []
    class_list = []
    pattern = re.compile(r'\w{2,10}?')

    with open(pospath,'r',encoding='utf-8') as f1:
        pos_texts = [line.strip() for line in f1.readlines()]
        # data_list +=pos_texts
        # class_list += [1]*len(pos_texts)
    with open(negpath,'r',encoding='utf-8') as f2:
        neg_texts = [line.strip() for line in f2.readlines()]
        # data_list +=neg_texts
        # class_list +=[0]*len(neg_texts)

    def da_biao_pian(texts, pattern, label):
        data_list = []
        class_list = []
        for text in texts:
            words_lst = jieba.lcut(text)
            words_lst_1 = []
            for word in words_lst:
                if pattern.match(word):
                    words_lst_1.append(word)
            data_list.append(words_lst_1)
            class_list.append(label)
        return data_list,class_list

    data_list_pos,class_list_pos = da_biao_pian(pos_texts,pattern,1)
    data_list_neg,class_list_neg = da_biao_pian(neg_texts,pattern,0)
    data_list = data_list_neg+data_list_pos
    class_list = class_list_neg+class_list_pos



    train_data_list, test_data_list, train_class_list, test_class_list = sklearn.model_selection.train_test_split(
        data_list, class_list, test_size=test_size)

    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            all_words_dict[word] = all_words_dict.get(word, 0) + 1

    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list

    # all_words_list = list(zip(*all_words_tuple_list)[0])
    all_words_list = [all_words_tuple_list[i][0] for i in range(len(all_words_tuple_list))]

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


# 按照版本选取文字材料
def text_get_by_version(pos_num = 600000,neg_num = 600000,test_size=0.2):
    data_list=[]
    class_list=[]
    with open(f'{now_version}/negs.txt','r',encoding='utf-8') as f:
        for i in range(neg_num):
            split_text = f.readline().split('\t')[-1]
            data_list.append(eval(f"[{split_text}]"))
            class_list.append(0)

    with open(f'{now_version}/poss.txt','r',encoding='utf-8') as f:
        for i in range(pos_num):
            split_text = f.readline().split('\t')[-1]
            data_list.append(eval(f"[{split_text}]"))
            class_list.append(1)

    train_data_list, test_data_list, train_class_list, test_class_list = sklearn.model_selection.train_test_split(
        data_list, class_list, test_size=test_size)

    all_words_dict = {}
    word_df = pd.read_excel(f'{now_version}/total_word_dict.xlsx',header=[0],index_col=[0])
    for i in range(len(word_df)):
        all_words_dict[word_df.loc[i,'word']]=word_df.loc[0,'fre']

    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)

    all_words_list = [all_words_tuple_list[i][0] for i in range(len(all_words_tuple_list))]

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

# 按需提取特征词
def words_dict(all_words_list, N, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(0, len(all_words_list) , 1):
        if n > N:  # feature_words的维度1000
            break

        if all_words_list[t] not in stopwords_set:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

# 文本特征加工
def text_features(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)

        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


# 训练与精度预测
def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## 使用nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy

# 文本预处理，分批 + 停用词库调入


# 模型训练
## 文本预处理


# 读取抽取同一步
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_get_by_version(neg_num=1000,pos_num=1000)

# 单抽取


# 生成stopwords_set
stopwords_file = f'{now_version}/stop_words.txt'
stopwords_set = make_word_set(stopwords_file)

## 文本特征提取和分类
# flag = 'nltk'
flag = 'sklearn'
Ns = range(1000, 2000, 20)
test_accuracy_list = []
i=0
for N in Ns:
    i+=1
    feature_words = words_dict(all_words_list, N, stopwords_set)
    train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words, flag)
    test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    test_accuracy_list.append(test_accuracy)
    if i%10==0:
        print(test_accuracy_list[-10:])

print(test_accuracy_list)

# 结果评价
#plt.figure()
plt.plot(Ns, test_accuracy_list)
plt.title('Relationship of number of words and test_accuracy')
plt.xlabel('number of words')
plt.ylabel('test_accuracy')
plt.show()
#plt.savefig('result.png')
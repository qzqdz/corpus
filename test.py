# 得到切片数据
import json
from collections import OrderedDict

import requests


def getSamepart(word1: str, word2: str) -> (int, int):
    n = len(word1)
    m = len(word2)

    dp = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if word1[i] == word2[j]:
                dp[i][j] = 1
                if i - 1 >= 0 and j - 1 >= 0:
                    dp[i - 1][j - 1] += 1

    max_len = 0
    start_at_first = 0
    for i in range(n):
        if dp[i][0] > max_len and dp[i][0] == min(n - i, m):
            max_len = dp[i][0]
            start_at_first = i
    for i in range(1, m):
        if dp[0][i] > max_len and dp[0][i] == min(n, m - i):
            max_len = dp[0][i]
            start_at_first = 0

    return max_len, start_at_first


# 聚类并得到索引

# 测试集如下
word_dic = {'就越重': {'a': 0.449474, 'r': 0.333333, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0},
            "一个网": {'a': 0.370612, 'r': 0.0, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0},
            '一个网页': {'a': 0.057632, 'r': 0.25, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0},
            '个网': {'a': 1.019895, 'r': 0.0, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0},
            "个网页": {'a': 0.32644, 'r': 0.333333, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0},
            '就越': {'a': 1.430859, 'r': 0.0, 'l': 0.0, 'f': 2, 'ns': 0.0, 's': 0.0}
            }


def FindClusters(word_dic):
    word_dic = sorted(word_dic.items(), key=lambda x: len(x[0]), reverse=True)
    word_dic = OrderedDict(word_dic)
    clusters_lst = [[k] for k in word_dic.keys()]

    words_dict = list(word_dic.items())
    words_num = range(len(clusters_lst))

    words_dict = dict(zip(words_num, words_dict))
    clusters_dict = dict(zip(words_num, clusters_lst))

    print(words_dict)
    for i in words_num:
        same_len = 0
        to_cluster_id = i
        for j in words_num:
            if j < i:
                continue

            if words_dict[i][0] != words_dict[j][0]:
                # new_same_len = 0
                # word_num = len(clusters_dict[j])
                #
                # for kwd in clusters_dict[j]:
                #     k_new_same_len, k_start = self.getSamepart(words_dict[i][0], kwd)
                #     new_same_len += k_new_same_len
                # new_same_len /= word_num
                new_same_len, start = getSamepart(words_dict[i][0], words_dict[j][0])
                if new_same_len > same_len:
                    same_len = new_same_len
                    to_cluster_id = j
                    clusters_dict[j].append(words_dict[i][0][:start] + words_dict[j][0])
        if to_cluster_id == i:
            continue

        clusters_dict[to_cluster_id].extend(clusters_dict[i])
        clusters_dict[i] = clusters_dict[i].remove(words_dict[i][0])

    clusters_dict = list(clusters_dict.values())
    clusters_lst = []
    for cluster in clusters_dict:
        if cluster is not None:
            clusters_lst.append(list(set(cluster)))

    return clusters_lst


kwlst = [['链接'], ['投票'], ['可以'], ['其他'], ['一种'], ['成都'],
         ['age', 'geR', 'eRank', 'ag', 'ageR', 'geRa', 'nk', 'Pa', 'Pag', 'Ra', 'geRan', 'an', 'Page', 'ge', 'Rank',
          'ageRa', 'eRa', 'PageR', 'Ran', 'eRan', 'ank', 'eR'], ['计算', '算法'], ['互联网', '联网', '互联'], ['思想'],
         ['的等', '的网页', '的网', '的来', '重要', '面的等', '页面的等', '来源', '要的信', '页面的', 'B页面', '面的', '页面', '的等级', '页面的等级', 'A页面',
          '要的', '面的等级', '网页', '的信', '等级'],
         ['Go', 'ogle', 'gl', 'ogl', 'oog', 'oogle', 'le', 'Goo', 'oo', 'gle', 'Googl', 'oogl', 'og', 'Goog'], ['A页'],
         ['B页'], ['的页面', '的页'], ['就是'], ['假设', '量假设', '量假', '质量'], ['个网', '个网页', '一个', '一个网', '一个网页'],
         ['越重', '就越', '就越重']]

summary = ["四川发文取缔全部不合规p2p。字节跳动与今日头条。成都日报，成都市，李太白与杜甫",
           "PageRank算法简介。",
           "是上世纪90年代末提出的一种计算网页权重的算法! ",
           "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。 ",
           "业界急需一种相对比较准确的网页重要性计算方法。 ",
           "是人们能够从海量互联网世界中找出自己需要的信息。 ",
           "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。 ",
           "Google把从A页面到B页面的链接解释为A页面给B页面投票。 ",
           "Google根据投票来源甚至来源的来源，即链接到A页面的页面。 ",
           "和投票目标的等级来决定新的等级。简单的说， ",
           "一个高等级的页面可以使其他低等级页面的等级提升。 ",
           "具体说来就是，PageRank有两个基本思想，也可以说是假设。 ",
           "即数量假设：一个网页被越多的其他页面链接，就越重）。 ",
           "质量假设：一个网页越是被高质量的网页链接，就越重要。 ",
           "总的来说就是一句话，从全局角度考虑，获取重要的信。 "]

text = '好像被消失了。。[疑问][疑问] //@造型师张亭亭:对呀，好奇怪， @作业本  本本去哪里了？[思考][疑问] //@高珑珂: 被河蟹了吗…[晕][晕][晕][晕][晕]'

api_key = 'oFbTU3GwhQkcxTdoRsCrNlEH'
secret_key = 'd0uLlkkpbtumeGCS2l6YFIRTtI3GCmYz'

sent_to_get_api_key = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}'

access_token_res = requests.post(sent_to_get_api_key).text

access_token = json.loads(access_token_res)['access_token']

data = {
    "text": "我爱祖国"
}
data = json.dumps(data)
res = requests.post(f'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={access_token}&charset=UTF-8', data=data)


# print(res)

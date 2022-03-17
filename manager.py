import jieba

from config import *
from emtion_classifer import Judge_By_Baidu
from find_word_and_fre import Find_new_words
from text_deal import Read_in, Into_backup_file, Update_corpus, Clear_words_dict

# 管理程序
print('读入文件')
summaries = Read_in()
neg_sens = []
negs = []
poss = []
mids = []



# 情感判断
i = 0
total_words_dict = {}
for summary in summaries:

    em_type, confidence = Judge_By_Baidu(summary)

    if confidence < confidence_rate:
        continue
    i += 1
    sen_cut = jieba.lcut(summary)

    if em_type == 0:
        negs.append(sen_cut)
        neg_sens.append(summary)
    elif em_type == 2:
        poss.append(sen_cut)
    else:
        mids.append(sen_cut)

    if i % 100 == 0:
        print(f"成功分类了{i}条评论！")

bad_words_dict = Find_new_words(neg_sens)
for sen_cut in poss + negs + mids:
    for word in sen_cut:
        total_words_dict[word] = total_words_dict.get(word, 0) + 1


Into_backup_file(poss, 1)
Into_backup_file(negs, 0)
Into_backup_file(total_words_dict, 3)
Into_backup_file(bad_words_dict, 4)


Update_corpus()
Clear_words_dict()

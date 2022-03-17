import os
import shutil
import time

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *


# 敏感词统计和合并
def mgc_he_bing():
    with open('./data/敏感词.txt', 'r', encoding='utf-8') as f:
        mg1_texts = f.readlines()
    with open('./data/sensitive.txt', 'r', encoding='utf-8') as f:
        mg2_texts = f.readlines()

    word_lst = []
    for line in mg1_texts:
        word_lst.append(line.strip())

    for line in mg2_texts:
        word_lst.append(line.strip())

    mg_dict = {}
    num = 0
    for wd in word_lst:
        mg_dict[wd] = mg_dict.get(wd, 0) + 1
        num += 1
        if num % 500 == 0:
            print(num)

    wd_lst = list(mg_dict.keys())

    fre_lst = list(mg_dict.values())

    mg_df = pd.DataFrame(columns=['word', 'fre'])
    mg_df['word'] = wd_lst
    mg_df['fre'] = fre_lst

    mg_df.to_excel('./data/mgck.xlsx')


# 处理情感分类的文件
def em_classier_file():
    with open('./data/neg60000.txt', 'r', encoding='utf-8') as f:
        texts1 = f.readlines()

    with open('./data/pos60000.txt', 'r', encoding='utf-8') as f:
        texts2 = f.readlines()

    negs = [text1.strip() for text1 in texts1]
    poss = [text2.strip() for text2 in texts2]

    with open('./data/Train2.txt', 'w', encoding='utf-8') as f:
        for neg in negs:
            f.write('0' + '\tneg\t\t' + neg + '\n')
        for pos in poss:
            f.write('1' + '\tpos\t' + pos + '\n')


# 将整个语料集合分成训练集和测试集
def IntoDifFile(data_list, class_list, mode='train'):
    output_df = pd.DataFrame()

    data_list = data_list.tolist()
    class_list = class_list.tolist()
    for i in range(len(data_list)):
        output_df.loc[i, 'query'] = '敏感舆情'
        output_df.loc[i, 'news'] = data_list[i]
        output_df.loc[i, 'relevance'] = class_list[i]

    output_df.to_csv(f'{DATA_FILE_PATH}/{mode}_data_list.tsv', sep='\t')


# 从combine中获取数据集
def huoqu_shujuji():
    sample = pd.read_excel(f'{DATA_FILE_PATH}/combinetest.xlsx', header=[0])

    # 长度处理
    sample['news_len'] = sample['news'].apply(lambda x: len(x))
    sample = sample[sample.news_len > 5]
    sample['news'] = sample['news'].apply(lambda x: x[:MAX_STR_LEN] if len(x) > MAX_STR_LEN else x)

    # 获取
    data_list = sample['news']
    class_list = sample['relevance']

    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(data_list, class_list,
                                                                                          test_size=30, shuffle=True)

    IntoDifFile(train_data_list, train_class_list, 'train')
    IntoDifFile(test_data_list, test_class_list, 'test')


def da_biao_pian(texts, pattern, emoji_pattern, label):
    data_list = []
    class_list = []
    i = 0

    for text in texts:

        text_1 = re.subn(u'(\/\/\@.*?\:|\@.*?\s)', '[@某人]', text)[0]
        words_lst_1 = emoji_pattern.findall(text_1)
        text_1 = re.subn(u'(\[.*?\])', ' ', text_1)[0]
        words_lst = jieba.lcut(text_1)

        for word in words_lst:
            if pattern.match(word):
                words_lst_1.append(word)

        data_list.append(words_lst_1)
        class_list.append(label)
        i += 1
        if i % 3000 == 0:
            print(f'进行了{i}次语句处理！')

    return data_list, class_list


# 生成情感语料库
def Make_corpus():
    with open(pospath, 'r', encoding='utf-8') as f1:
        pos_texts = [line.strip() for line in f1.readlines()]
        # data_list +=pos_texts
        # class_list += [1]*len(pos_texts)
    with open(negpath, 'r', encoding='utf-8') as f2:
        neg_texts = [line.strip() for line in f2.readlines()]
        # data_list +=neg_texts
        # class_list +=[0]*len(neg_texts)

    if not os.path.exists(corpath):
        os.mkdir(corpath)

    version = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

    data_list_pos, _ = da_biao_pian(pos_texts, pattern, emoji_pattern, 1)
    data_list_neg, _ = da_biao_pian(neg_texts, pattern, emoji_pattern, 0)

    if not os.path.exists(f'{corpath}/{version}'):
        os.mkdir(f'{corpath}/{version}')

    in_to_file = f'{corpath}/{version}/'

    with open(f'{in_to_file}{file_type_list[1]}', 'w', encoding='utf-8') as f:
        for pos_lst in data_list_pos:
            f.write(f'{"1"}\t{"pos"}\t\t{str(pos_lst).strip("[]")}\n')

    with open(f'{in_to_file}{file_type_list[0]}', 'w', encoding='utf-8') as f:
        for neg_lst in data_list_neg:
            f.write(f'{"0"}\t{"neg"}\t\t{str(neg_lst).strip("[]")}\n')

    combine_lst = data_list_pos + data_list_neg
    total_word_dict = {}
    num = 0
    for text_lst in combine_lst:
        for word in text_lst:
            total_word_dict[word] = total_word_dict.get(word, 0) + 1
            num += 1
            if num % 500 == 0:
                print(num)

        wd_lst = list(total_word_dict.keys())

        fre_lst = list(total_word_dict.values())

        mg_df = pd.DataFrame(columns=['word', 'fre'])
        mg_df['word'] = wd_lst
        mg_df['fre'] = fre_lst

        mg_df.to_excel(f'{in_to_file}{file_type_list[3]}')

    shutil.copyfile('./data/stop_words.txt', f'{in_to_file}{file_type_list[2]}')
    shutil.copyfile('./data/mgck.xlsx', f'{in_to_file}{file_type_list[4]}')

    with open('config.py', 'r', encoding='utf-8') as f:
        text = f.read()
        try:
            text = text.replace(f"now_version = '{now_version}'", f"now_version = '{in_to_file[:-1]}'")
        except:
            text = text + f"""\n# 当前版本语料库\nnow_version = '{in_to_file[:-1]}'"""
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(text)


# 更新语料库
# 将备份写入语料库
def Update_corpus():
    file_lst = os.listdir(now_version)
    backup_file_lst = []
    for file in file_lst:
        if file.split('_')[0] == 'backup':
            backup_file_lst.append(file)
    if 'total' in ' '.join(backup_file_lst) or 'mg' in ' '.join(backup_file_lst):
        with open(f'{now_version}/{file_type_list[2]}', 'r', encoding='utf-8') as f:
            stop_words = list(set(line.strip() for line in f.readlines()))

    for backup_file in backup_file_lst:

        if backup_file.split('_')[-1] == file_type_list[0]:
            with open(f'{now_version}/{file_type_list[0]}', 'r', encoding='utf-8') as f:
                with open(f'{now_version}/{backup_file}', 'r', encoding='utf-8') as of:
                    stop_words = list(set([line.strip() for line in f.readlines()] + [line.strip() for line in of.readlines()]))

            with open(f'{now_version}/{file_type_list[0]}', 'w', encoding='utf-8') as f:
                for word in stop_words:
                    f.write(f'{word}\n')

        if backup_file.split('_')[-1] == file_type_list[1]:
            with open(f'{now_version}/{file_type_list[1]}', 'r', encoding='utf-8') as f:
                with open(f'{now_version}/{backup_file}', 'r', encoding='utf-8') as of:
                    stop_words = list(set([line.strip() for line in f.readlines()] + [line.strip() for line in of.readlines()]))

            with open(f'{now_version}/{file_type_list[1]}', 'w', encoding='utf-8') as f:
                for word in stop_words:
                    f.write(f'{word}\n')

        if backup_file.split('_')[-1] == file_type_list[2]:
            with open(f'{now_version}/{file_type_list[2]}', 'r', encoding='utf-8') as f:
                with open(f'{now_version}/{backup_file}', 'r', encoding='utf-8') as of:
                    stop_words = list(set([line.strip() for line in f.readlines()] + [line.strip() for line in of.readlines()]))

            with open(f'{now_version}/{file_type_list[2]}', 'w', encoding='utf-8') as f:
                for word in stop_words:
                    f.write(f'{word}\n')

        if backup_file[7:] == file_type_list[3]:

            out_df = pd.read_excel(f'{now_version}/{backup_file}', index_col=[0], header=[0])
            in_df = pd.read_excel(f'{now_version}/{file_type_list[3]}', index_col=[0], header=[0])

            tail = len(in_df)
            for i in range(len(out_df)):
                if out_df.loc[i, 'word'] in stop_words:
                    continue
                if out_df.loc[i, 'word'] not in in_df['word'].tolist():
                    in_df.loc[tail, 'word'] = out_df.loc[i, 'word']
                    in_df.loc[tail, 'fre'] = out_df.loc[i, 'fre']
                    tail += 1
                else:
                    in_df.loc[in_df['word'] == out_df.loc[i, 'word'], 'fre'] += out_df.loc[i, 'fre']

            in_df.to_excel(f'{now_version}/{file_type_list[3]}')


        if backup_file[7:] == file_type_list[4]:

            out_df = pd.read_excel(f'{now_version}/{backup_file}', index_col=[0], header=[0])
            in_df = pd.read_excel(f'{now_version}/{file_type_list[4]}', index_col=[0], header=[0])

            tail = len(in_df)
            for i in range(len(out_df)):
                if out_df.loc[i, 'word'] in stop_words:
                    continue

                if str(out_df.loc[i, 'word']) not in in_df['word'].tolist():
                    in_df.loc[tail, 'word'] = out_df.loc[i, 'word']
                    in_df.loc[tail, 'fre'] = out_df.loc[i, 'fre']
                    tail += 1
                else:
                    in_df.loc[in_df['word'] == out_df.loc[i, 'word'], 'fre'] += out_df.loc[i, 'fre']

            in_df.to_excel(f'{now_version}/{file_type_list[4]}')

        os.remove(f'{now_version}/{backup_file}')


# 迁移情感语料库
def Migrate_corpus(negs=None, poss=None, words_dict=None, bad_words_dict=None, stop_words=None):

# def Migrate_corpus():
#     negs = [['我', '讨厌', 'zg']]
#     poss = [['我', '爱', '中国']]
#     neg_ori = ['我讨厌zg']
#     poss_ori = ['我爱中国']
#     words_dict = {'dp': 10, '台独': 5}
#     bad_words_dict = {'dp': 10, '台独': 5}
#     stop_words = [';l']
    version = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    new_corpath = f'{corpath}/{version}/'

    os.mkdir(new_corpath[:-1])
    for file in os.listdir(now_version):
        old_file_path = f'{now_version}/{file}'
        new_file_path = f'{new_corpath}{file}'
        shutil.copyfile(old_file_path, new_file_path)
    with open('config.py', 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace(f"now_version = '{now_version}'", f"now_version = '{new_corpath[:-1]}'")

    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(text)

    if negs:
        with open(f'{new_corpath[:-1]}/backup_{file_type_list[0]}', 'a+', encoding='utf-8') as f:
            for sen in negs:
                f.write(f'0\tneg\t\t{str(sen).strip("[]")}\n')

        shutil.copyfile(f'{now_version}/backup_{file_type_list[0]}',f'{check_path}/backup_{file_type_list[0]}')
    if poss:
        with open(f'{new_corpath[:-1]}/backup_{file_type_list[1]}', 'a+', encoding='utf-8') as f:
            for sen in poss:
                f.write(f'1\tpos\t\t{str(sen).strip("[]")}\n')

        shutil.copyfile(f'{now_version}/backup_{file_type_list[1]}', f'{check_path}/backup_{file_type_list[1]}')

    if stop_words:
        with open(f'{new_corpath[:-1]}/backup_{file_type_list[2]}', 'a+', encoding='utf-8') as f:
            for word in stop_words:
                f.write(f'{word}\n')

        shutil.copyfile(f'{now_version}/backup_{file_type_list[2]}', f'{check_path}/backup_{file_type_list[2]}')

    if words_dict:
        try:
            df = pd.read_excel(f'{new_corpath[:-1]}/backup_{file_type_list[3]}', index_col=[0], header=[0])
        except:
            df = pd.DataFrame(columns=['word', 'fre'])
        tail = len(df)
        for k, v in words_dict.items():
            if df.loc[df['word'] == k, 'fre'].empty:
                df.loc[tail, 'word'] = k
                df.loc[tail, 'fre'] = v
                tail += 1
            else:
                df.loc[df['word'] == k, 'fre'] += v
        df.to_excel(f'{new_corpath[:-1]}/backup_{file_type_list[3]}')

        shutil.copyfile(f'{now_version}/backup_{file_type_list[3]}', f'{check_path}/backup_{file_type_list[3]}')
    if bad_words_dict:
        try:
            df = pd.read_excel(f'{new_corpath[:-1]}/backup_{file_type_list[4]}', index_col=[0], header=[0])
        except:
            df = pd.DataFrame(columns=['word', 'fre'])
        tail = len(df)
        for k, v in bad_words_dict.items():
            if df.loc[df['word'] == k, 'fre'].empty:
                df.loc[tail, 'word'] = k
                df.loc[tail, 'fre'] = v
                tail += 1
            else:
                df.loc[df['word'] == k, 'fre'] += v
        df.to_excel(f'{new_corpath[:-1]}/backup_{file_type_list[4]}')
        shutil.copyfile(f'{now_version}/backup_{file_type_list[4]}', f'{check_path}/backup_{file_type_list[4]}')

    return new_corpath[:-1]


# 删除情感语料库
def Del_corpus():
    all_version = os.listdir(corpath)
    print(f'当前版本有:{str(all_version).strip("[]")}')
    del_version = input('输入需要删除的文件版本：\n')
    if len(all_version) > 1 and del_version in all_version:
        del all_version[all_version.index(del_version)]
    elif len(all_version) <= 1:
        print('版本过少，无法删除\n')
    else:
        print('输入错误\n')
        return

    versions = [int(''.join(version.split('_'))) for version in all_version]

    del_version = f'{corpath}/{del_version}'
    if now_version == del_version:
        # try:
        latest_version = str(max(versions))
        latest_version = f'{latest_version[:4]}_{latest_version[4:6]}_{latest_version[6:8]}_{latest_version[8:10]}_{latest_version[10:12]}_{latest_version[12:]}'
        new_version = '/'.join(now_version.split('/')[:-1]) + '/'
        new_version += latest_version
        shutil.rmtree(del_version)
        with open('config.py', 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace(f"now_version = '{now_version}'", f"now_version = '{new_version}'")

        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(text)
        # except:
        #     print("删除失败")
    else:
        try:
            shutil.rmtree(del_version)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


# 选择语料库版本
def Choice_corpus():
    all_version = os.listdir(corpath)
    print(f'当前版本有:{str(all_version).strip("[]")}')
    del_version = input('输入想要使用的文件版本：\n')
    if del_version in all_version:
        del all_version[all_version.index(del_version)]
    else:
        print('输入错误\n')
        return

    versions = [int(''.join(version.split('_'))) for version in all_version]
    latest_version = str(max(versions))
    latest_version = f'{latest_version[:4]}_{latest_version[4:6]}_{latest_version[6:8]}_{latest_version[8:10]}_{latest_version[10:12]}_{latest_version[12:]}'
    new_version = '/'.join(now_version.split('/')[:-1]) + '/'
    new_version += latest_version

    with open('config.py', 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace(f"now_version = '{now_version}'", f"now_version = '{new_version}'")

    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(text)


# 读入新材料
def Read_in():
    all_file = os.listdir(summary_path)
    material = input(f"选择新的输入材料：{','.join(all_file)}")
    if material not in all_file:
        print('输入错误')
        return
    summary_df = pd.read_excel(f'{summary_path}/{material}', header=[0], index_col=[0])
    summaries = list(summary_df['summary'])
    summaries_1 = []
    for summary in summaries:
        try:
            text_1 = re.subn(u'(\/\/\@.*?\:|\@.*?\s)', '[@某人]', summary)[0]
            summaries_1.append(text_1)
        except:
            pass
    return summaries_1


# 加入语料库
def Into_backup_file(summaries, file_type_id):
    if file_type_id <= 2:
        summaries: list
    else:
        summaries: dict
    if file_type_id == 0 or file_type_id == 1:
        with open(f'{now_version}/backup_{file_type_list[file_type_id]}', 'a+', encoding='utf-8') as f:
            for summary in summaries:
                f.write(f'{file_type_id}\t{file_type_list[file_type_id][:3]}\t\t{str(summary).strip("[]")}\n')

    if file_type_id == 2:
        with open(f'{now_version}/backup_{file_type_list[file_type_id]}', 'a+', encoding='utf-8') as f:
            for summary in summaries:
                f.write(f'{summary}\n')

    if file_type_id == 3 or file_type_id == 4:

        words_df = pd.DataFrame(columns=['word', 'fre'])
        row = 0
        for k, v in summaries.items():
            words_df.loc[row, 'word'] = k
            words_df.loc[row, 'fre'] = v
            row += 1

        words_df.to_excel(f'{now_version}/backup_{file_type_list[file_type_id]}')


def Move_into_file():
    files = os.listdir(check_path)

    if files:
        for file in files:
            if file.split('_')[-1] in file_type_list:

                old_path = f'{check_path}/{file}'
                new_path = f'{now_version}/backup_{file.split("_")[-1]}'
                shutil.copyfile(old_path, new_path)
                os.remove(f'{check_path}/{file}')
            else:
                print(f"{file}文件无法同步，请重命名文件")

    else:
        print('无更新文件')


def Clear_words_dict():
    j=0
    in_lst = []
    with open(f"{now_version}/{file_type_list[2]}",'r',encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    for k in range(3,5):
        out_df = pd.read_excel(f'{now_version}/{file_type_list[k]}')
        in_df = pd.DataFrame(columns=['word', 'fre'])
        for i in range(len(out_df)):
            if out_df.loc[i,'word'] in stop_words:
                continue
            if out_df.loc[i,'word'] in in_lst:
                in_df.loc[in_df['word']==out_df.loc[i,'word'], 'fre'] += out_df.loc[i, 'fre']
                continue
            if out_df.loc[i,'word'].isdigit():
                continue
            in_df.loc[j,'word'] = out_df.loc[i,'word']
            in_df.loc[j, 'fre'] = out_df.loc[i, 'fre']
            in_lst.append(out_df.loc[i, 'word'])
            j+=1
            if j%1000 ==0:
                print(f"成功载入了{j}条")

        in_df.to_excel(f'{now_version}/{file_type_list[k]}')




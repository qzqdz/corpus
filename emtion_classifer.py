import json
import time

import requests
import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

from config import *


def Judge_by_BERT():
    target = ['neg','pos']

    cls_model = "./data/model/bert_multi_classify_model.bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = "./data/bert/vocab.txt"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    # 加载字典
    word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
    tokenizer = Tokenizer(word2idx)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name, model_class="cls", target_size=len(target))
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=cls_model, device=device)
    test_data = ["[怒] //@左小祖咒:[话筒] //@ziyewong://@靠江魏三: //@明可mk://@作家-天佑: 天杀的",
                "从来都要求带着镣铐跳舞。。。",
                "一流的经纪公司是超模的摇篮！",
                '各位都是好声音啊，演唱会就星外星筹办了',
                 '良心泯灭啊！！！[怒][怒] [怒]',
                 '[泪]我是小清新 //@Yihwei_Towne:以撒大的性格放卫生纸应该差不多。']
    for text in test_data:
        with torch.no_grad():
            text, text_ids = tokenizer.encode(text)
            text = torch.tensor(text, device=device).view(1, -1)
            print(target[torch.argmax(bert_model(text)).item()])



def Judge_By_Baidu(string:str)->(int,float):
    data = {
        "text": string
    }
    data = json.dumps(data)
    while True:
        res = requests.post(f'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={access_token}&charset=UTF-8', data=data)
        res_dict = json.loads(res.text)
        if res.status_code==200 and 'items' in list(res_dict.keys()):
            break
        else:
            time.sleep(0.5)

    res_dict = res_dict['items'][0]



    return int(res_dict['sentiment']),float(res_dict['confidence'])

import re

pattern = re.compile(r'\w{2,10}?')
emoji_pattern = u'(\[.*?\])'


test_size = 0.2
train_size = 0.8
MAX_STR_LEN = 450
DATA_FILE_PATH = './data'

target = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]
vocab_path = "./data/bert/vocab.txt"  # 模型字典的位置
PRETRAINED_MIDEL_NAME_2 = "roberta"
model_path = "./data/model/seq_bert_ner_model.bin"

PRETRAINED_MODEL_NAME_1 = "bert-base-chinese"
NUM_LABELS = 2
BATCH_SIZE = 5

cuda_use = True


# 情感分析模型
# 获取access_token
# sent_to_get_api_key = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}'
# access_token_res = requests.post(sent_to_get_api_key).text
# access_token = json.loads(access_token_res)['access_token']

access_token = '24.45ab0949848f7e4e44432b9201f73678.2592000.1649927884.282335-25767620'
api_key = 'oFbTU3GwhQkcxTdoRsCrNlEH'
secret_key = 'd0uLlkkpbtumeGCS2l6YFIRTtI3GCmYz'

'{"log_id": 1948138297255636847, "text": "在食品卫生安全这一方面，法律并不完善，真的是处罚太轻、犯罪成本太低，已经类似于鼓励犯罪了", "items": [{"positive_prob": 0.00387671, "confidence": 0.991385, "negative_prob": 0.996123, "sentiment": 0}]}'
'{"log_id": 7887309081195943599, "text": "大概热度过了，这校长要被自动离职了", "items": [{"positive_prob": 0.00249592, "confidence": 0.994453, "negative_prob": 0.997504, "sentiment": 0}]}'

emtion = ['neg','mid','pos']
confidence_rate = 0.6

# 句子敏感程度：入选比率
sentences_rate = 0.6

# 单词检测敏感程度
word_rate = 0.4

# 语料库文件类型
file_type_list = ['negs.txt', 'poss.txt', 'stop_words.txt', 'total_word_dict.xlsx', 'mgck.xlsx']


# 预警模型
pospath = './data/pos60000.txt'
negpath = './data/neg60000.txt'

# 新材料输入库
summary_path = './summary'

# 人工检查语料库
check_path ='./check_file'

# 核心语料库,以下请勿乱动
# 2022_03_14_21_23_51
corpath = './corpora'
now_version = './corpora/2022_03_16_22_04_43'

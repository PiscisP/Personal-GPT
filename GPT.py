import os
import tiktoken
from convokit import Corpus, download
import torch.nn as nn
import torch



#declare hyperparameters
batch_size = 4
context_length = 100
d_model = 64
num_blocks = 8
num_heads = 4
learning_rate = 0.001
dropout = 0.1
max_iterations = 1000
evel_interval = 100
ever_iteration = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)



# 指定语料库存放的路径
corpus_path = 'C:/Users/Ranshao/.convokit/downloads/movie-corpus'

# 加载或下载语料库
if os.path.exists(corpus_path):
    corpus = Corpus(filename=corpus_path)
else:
    print("Corpus path does not exist. Downloading corpus...")
    # 下载电影语料库
    corpus_path = download('movie-corpus', data_dir='C:/Users/Ranshao/.convokit/downloads')
    corpus = Corpus(filename=corpus_path)

# 存储对话文本的列表
conversations_texts = []

# 限制处理的对话数量
max_conversations = 5000



'''
print("总话语数:", len(corpus.utterances))
print("总对话数（会话数）:", len(corpus.conversations))


for conversation_id in corpus.conversations:
    conversation = corpus.get_conversation(conversation_id)
    print(f"对话ID: {conversation_id}")
    for utterance in conversation.iter_utterances():
        print(f"{utterance.speaker.id}: {utterance.text}")
    break  
'''



# 遍历语料库中的对话
for i, conversation_id in enumerate(corpus.conversations):
    if i >= max_conversations:
        break
    conversation = corpus.get_conversation(conversation_id)
    # 合并对话中的所有话语成一个字符串
    conversation_text = ' '.join([utterance.text for utterance in conversation.iter_utterances()])
    conversations_texts.append(conversation_text)


# toekenize
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode("".join(conversations_texts))
print(len(tokenized_text))
max_token_value = max(tokenized_text)
print(max_token_value)

#split into train and test dataset
train_size = int(0.9 * len(tokenized_text))
train_dataset = tokenized_text[:train_size]
test_dataset = tokenized_text[train_size:]

#FFN Layer
'''
Two Fully Connected Layers with ReLU Activation Function
'''
class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model):
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self. relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout = 0.1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x
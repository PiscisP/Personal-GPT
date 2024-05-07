import os
import tiktoken
from convokit import Corpus, download
import torch.nn as nn
import torch
from torch.nn import functional as F



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
class Feedforward(nn.Module):
    def __init__(self, d_model):
        super(Feedforward, self).__init__()
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
    
class Head(nn.Module):
    '''
    batch_size = 4
    context_size = 100
    d_model = 64
    '''
    def _init_ (self):
        super(self)._init_()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 64 * 64
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        print(Q.shape, K.shape, V.shape)
        attention = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attention = attention.masked_fill(self.mask == 0, float('-inf'))
        attention = nn.Softmax(attention, dim = -1)
        attention = self.dropout(attention)
        output = attention @ V

        return output

class MultiHeadAttention(nn.Module):
    def _init_(self):
        super(self)._init_()
        #循环调用单头注意力
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        #全连接层
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        out = self.dropout(self.proj(heads))
        return out

class Block(nn.Module):
    def _init_(self):
        super(self)._init_()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.multihead = MultiHeadAttention()
        self.ffn = Feedforward(d_model)

    def forward(self, x):
        x = x + self.multihead(self.ln1(x))
        x = x + self.ffn(self.ffn(self.ln2(x)))
        return x
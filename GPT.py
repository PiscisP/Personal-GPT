import os
from convokit import Corpus

# 指定语料库存放的路径
corpus_path = 'C:/Users/Ranshao/.convokit/downloads/movie-corpus'

# 加载语料库
if os.path.exists(corpus_path):
    corpus = Corpus(filename=corpus_path)
else:
    print("Corpus path does not exist.")

# 存储对话文本的列表
conversations_texts = []

# 限制处理的对话数量
max_conversations = 10000

# 遍历语料库中的对话
for i, conversation_id in enumerate(corpus.conversations):
    if i >= max_conversations:
        break
    conversation = corpus.get_conversation(conversation_id)
    # 合并对话中的所有话语成一个字符串
    conversation_text = ' '.join([utterance.text for utterance in conversation.iter_utterances()])
    conversations_texts.append(conversation_text)

# 现在 conversations_texts 包含了最多 10000 个对话，每个对话都是一个长字符串
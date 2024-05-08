import os
import tiktoken
import torch.nn as nn
import torch
from torch.nn import functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def check_tensor(tensor, name="Tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"{name} contains NaN or Inf values.")


#declare hyperparameters
batch_size = 16
context_length = 200
d_model = 128
num_heads = 8
learning_rate = 0.001
dropout = 0.1
max_iterations = 10000
evel_interval = 200
ever_iteration = 100
n_layer = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)



# 指定语料库存放的路径
corpus_path = 'C:/Users/11632/OneDrive/桌面/GPT/Pillar of the Earth.txt'

# Load the contents of the file
with open(corpus_path, 'r', encoding='utf-8') as file:
    corpus = file.read()

# toekenize
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(corpus)
print(len(tokenized_text))
max_token_value = max(tokenized_text) + 1
print(max_token_value)

data = torch.tensor(tokenized_text, dtype=torch.long, device=device)
#split into train and test dataset
train_size = int(0.9 * len(data))
train_dataset = data[:train_size]
val_dataset = data[train_size:]

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
        self.dropout = nn.Dropout(dropout)


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
    def __init__(self,num_heads,head_size):
        super().__init__()
        # 64 * 64
        self.Wq = nn.Linear(d_model, head_size,bias = False)
        self.Wk = nn.Linear(d_model, head_size,bias = False)
        self.Wv = nn.Linear(d_model, head_size,bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        assert T <= context_length
        assert C == d_model
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        weight = Q @ K.transpose(-2, -1) * K.shape[-1]**-0.5

        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        output = weight @ V
        #print("Q shape:", Q.shape)
        #print("K shape:", K.shape)
        #print("Output of transpose operation:", K.transpose(-2, -1).shape)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        #循环调用单头注意力
        self.heads = nn.ModuleList([Head(num_heads, head_size) for _ in range(num_heads)])
        #全连接层
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.multihead = MultiHeadAttention(num_heads, head_size)
        self.ffn = Feedforward(d_model)

    def forward(self, x):
        x = x + self.multihead(self.ln1(x))
        x = x + self.ffn(self.ffn(self.ln2(x)))
        return x

#define model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(max_token_value, d_model)
        self.position_embedding_table = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, max_token_value)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx , targets = None):
        '''
        idx: [B, T] tensor of token indices
        B: batch_size = 4
        T: context_length = 100
        C: d_model = 64
        max_token_value = max(tokenized_text)
        '''
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        #(batch_size,context_length,max_token_value)
        logits = self.fc(x)

        if targets is None:
            loss = None
        else:
            #B, T, max_token_value = logits.shape
            logits = logits.view(-1 ,  max_token_value)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_dataset if split == 'train' else val_dataset
    ix = torch.randint(len(data) - context_length, (batch_size,))
    # Ensure data is a tensor, convert if necessary:
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long, device=device)

    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = GPT().to(device)

@torch.no_grad()
def estimate_loss(model, split, ever_iteration, device):
    losses = torch.zeros(ever_iteration, device=device)
    model.eval()
    for k in range(ever_iteration):
        X, Y = get_batch(split)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()



def load_model(model_path, device):
    model = GPT().to(device)  # Initialize the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved state
    model.eval()  # Set the model to evaluation mode
    return model

def generate_text(input_text, model_path='model.pth', max_new_tokens=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    encoding = tiktoken.get_encoding("cl100k_base")  # Ensure this uses the correct encoding scheme
    start_ids = encoding.encode(input_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = model.generate(x, max_new_tokens)
    generated_text = encoding.decode(generated_ids[0].tolist())
    
    return generated_text


def train_model(model, max_iterations, eval_interval, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    tracked_losses = {'train': [], 'val': []}
    for iteration in range(max_iterations):
        xb, yb = get_batch('train')
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if iteration % eval_interval == 0 or iteration == max_iterations - 1:
            train_loss = estimate_loss(model, 'train', ever_iteration, device)
            val_loss = estimate_loss(model, 'val', ever_iteration, device)
            tracked_losses['train'].append(train_loss)
            tracked_losses['val'].append(val_loss)
            print(f"Step {iteration}: Train Loss {train_loss:.4f}, Validation Loss {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved to", save_path)
    return tracked_losses


def main():
    action = "generate"
    if action == "train":
        model = GPT().to(device)
        train_model(model, max_iterations, evel_interval, 'model.pth')
    elif action == "generate":
        start_text = "大教堂完工了。"
        generated_text = generate_text(start_text, 'model.pth', 100)
        print("Generated text:")
        print(generated_text)



if __name__ == '__main__':
    main()
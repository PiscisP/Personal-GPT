import os
import torch.nn as nn
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def check_tensor(tensor, name="Tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"{name} contains NaN or Inf values.")

# 设置超参数
batch_size = 16
sequence_length = 30
d_model = 256
num_heads = 8
learning_rate = 1e-3
dropout = 0.1
n_layer = 6
dim_feedforward = 512

num_features = 9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

def load_data(corpus_path):
    corpus_df = pd.read_csv(corpus_path)
    corpus_df['date'] = pd.to_datetime(corpus_df['date'], format='%Y/%m/%d %H:%M')
    corpus_df.set_index('date', inplace=True)
    data_values = corpus_df.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    return data_scaled, scaler, corpus_df.columns, corpus_df.index

# Load and scale your data
data_scaled, scaler, feature_names, data_index = load_data('C:/Users/11632/OneDrive/桌面/GPT/sequence_test/water.csv')

# Convert scaled data back to DataFrame for plotting
scaled_df = pd.DataFrame(data_scaled, columns=feature_names, index=data_index)
'''
# Plotting each feature
plt.figure(figsize=(12, 15))
for i, feature in enumerate(feature_names):
    plt.subplot(len(feature_names), 1, i + 1)
    plt.plot(scaled_df.index, scaled_df[feature])
    plt.title(feature)
    plt.tight_layout()
'''
plt.show()

# 创建输入序列和目标序列
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


corpus_path = 'C:/Users/11632/OneDrive/桌面/GPT/sequence_test/water.csv'
data_np, scaler, column, index = load_data(corpus_path)
x, y = create_sequences(data_np, sequence_length)
X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_tensor.shape, y_tensor.shape)

# Create the full dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Calculate sizes for train, validation, and test sets
total_size = len(dataset)
print("Total dataset size:", len(dataset))
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size 
print("Train size:", train_size)
print("Validation size:", val_size)
print("Test size:", test_size)

# Create new TensorDataset instances for train, validation, and test sets
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
val_dataset = TensorDataset(X_tensor[train_size:train_size + val_size], y_tensor[train_size:train_size + val_size])
test_dataset = TensorDataset(X_tensor[train_size + val_size:], y_tensor[train_size + val_size:])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Train Loader Size:", len(train_loader))
print("Validation Loader Size:", len(val_loader))
print("Test Loader Size:", len(test_loader))

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

        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        assert T <= sequence_length
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
        #recursively create num_heads heads
        self.heads = nn.ModuleList([Head(num_heads, head_size) for _ in range(num_heads)])
        #fc layer
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, sequence_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#define model
class Transformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, d_model)
        self.position_embeddings = PositionalEncoding(d_model, sequence_length)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_features)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        '''
        idx: [B, T] tensor of token indices, where B is batch size and T is sequence length.
        positions: [B, T] tensor of position indices corresponding to each token.
        '''
        #print("Shape of idx:", x.shape)
        B, T, F = x.shape

        x = self.feature_projection(x)  # Now [B, T, d_model]n
        x = self.position_embeddings(x)  # [B, T, d_model]
        
        # Step 2: Pass the combined embeddings through each of the transformer blocks
        x = self.blocks(x)  # Sequentially apply each Block
        
        # Step 3: Apply the final layer norm
        x = self.ln(x)  # [B, T, d_model]
        
        # Step 4 [B, d_model] -> [B, num_features]
        x = self.fc(x[:, -1, :])
        
        return x

def test_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Load model
    model = Transformers().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    # Perform prediction
    all_predictions = []
    all_actuals = []  # To store actual outputs for comparison
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())  # Append predictions directly
            all_actuals.append(targets.cpu().numpy())  # Append actual targets for loss calculation

    # Convert lists to numpy arrays for easier manipulation
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    mse_losses = (all_predictions - all_actuals) ** 2
    # Plot loss for each feature across time steps
    plt.figure(figsize=(14, 10))
    num_features = all_predictions.shape[1]
    for i in range(num_features):
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(mse_losses[:, i], label=f'Feature {i+1} MSE Loss', marker='o', linestyle='-')
        plt.title(f'Feature {i+1} MSE Loss Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()

num_epochs = len(train_dataset) // batch_size
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stop_count = 0
    min_val_loss = float('inf')
    scheduler = ExponentialLR(optimizer, gamma=0.95)  # 设置衰减率为0.95
    
    # 用于存储损失记录
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        batch_train_losses = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            output = model(inputs)

            # Compute loss
            loss = criterion(output, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        epoch_train_loss = np.mean(batch_train_losses)
        train_losses.append(epoch_train_loss)
        scheduler.step()  # 更新学习率

        # Validation phase
        model.eval()
        batch_val_losses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                batch_val_losses.append(loss.item())

        epoch_val_loss = np.mean(batch_val_losses)
        val_losses.append(epoch_val_loss)
        
        # Early stopping logic could be applied here
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), save_path)  # Save the best model
        else:
            early_stop_count += 1
        

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

        # Uncomment below if you have implemented early stopping
        # if early_stop_count >= 20:
        #     print("Early stopping!")
        #     break

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    action = "generate"
    #input_dim = data_np.shape[1]  # 定义输入维度

    if action == 'train':
        model = Transformers().to(device)
        train_model(model, train_loader, val_loader, num_epochs, learning_rate, 'model.pth')
    
    elif action == 'generate':
        test_model()
    else:
        print("Invalid action. Please enter 'train' or 'generate'.")

if __name__ == '__main__':
    main()
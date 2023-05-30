import time
import torch
from tqdm import trange
from utils import load_data
from torch.optim import Adam
from model import cag_gpt_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 2
batch_size = 8
lr = 1e-5

data = load_data('./data/train.txt', batch_size)
train_data = data[:int(0.8*len(data))]
train_data_size = batch_size * len(train_data)
valid_data = data[int(0.8*len(data)):]
valid_data_size = batch_size * len(valid_data)

model = cag_gpt_model(device)
model.from_pretrained('gpt2')
optimizer = Adam(model.parameters(), lr = lr)


begin = time.time()
print('is training model ...')
for epoch in trange(epochs, desc = 'Epoch'):
    this_epoch_loss = 0
    model.train()
    for i in trange(len(train_data), desc = 'Iteration'):
        loss = 0
        for j in range(batch_size):
            topic, source, target = train_data[i][j]
            loss += model(topic, source, target)
        this_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.zero_grad()
    this_epoch_loss = this_epoch_loss/train_data_size
    print('\n train loss:{:.2f}'.format(this_epoch_loss))

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i in range(len(valid_data)):
            loss = 0
            for j in range(batch_size):
                topic, source, target = valid_data[i][j]
                loss += model(topic, source, target)
            valid_loss += loss.item()
        valid_loss = valid_loss/valid_data_size
        print('\n valid loss:{:.2f}'.format(valid_loss))

end = time.time()
print('time using:{}'.format(end - begin))
#模型保存
model.save_pretrained('model')

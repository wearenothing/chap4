import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


# 1. Download Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')



# print(train_data.shape)
# print(test_data.shape)
#
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 2. Data Preprocessing
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# Normalize numerical features
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# Process non-numerical features
all_features = pd.get_dummies(all_features, dummy_na=True)

# 3. Read data

# separate all_features into train_features and test_features, then separate train_features into X_train and X_valid

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 4. Train
loss = nn.MSELoss()
in_features = test_features.shape[1]

def getNet():
    # net = nn.Sequential(nn.Linear(in_features,1))
    net = nn.Sequential(nn.Linear(in_features,256),nn.ReLU(),nn.Linear(256,10),nn.ReLU(), nn.Linear(10,1))
    # def init_weight(m):
    #     if type(m)== nn.Linear:
    #         nn.init.normal_(m.weight,std=0.1)
    # net.apply(init_weight)
    return net

def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rates, weight_decay, batch_size,):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels),batch_size)
    trainer = torch.optim.Adam(net.parameters(),lr=learning_rates,weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            trainer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        ind = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part= X[ind,:], y[ind]
        if j==i:
            X_valid, y_valid = X_part, y_part
        elif X_train is not None:
            X_train = torch.cat((X_train,X_part),0)
            y_train = torch.cat((y_train,y_part),0)
        else:
            X_train, y_train = X_part, y_part

    return X_train, y_train, X_valid, y_valid

def k_fold(k,X_data,y_data,num_epochs, learning_rate, weight_decay, batch_size):

    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k,i,X_data,y_data)
        net =getNet()
        train_l, valid_l = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_l[-1]
        valid_l_sum += valid_l[-1]
        if i==0:
            d2l.plot(list(range(1,num_epochs+1)),[train_l, valid_l],xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_l[-1]):f}, 'f'valid log rmse {float(valid_l[-1]):f}')
    return train_l_sum/k, valid_l_sum/k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.03, 0, 64
train_l, valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f} ',f'avg valid log rmse: {float(valid_l): f}')

# Now we get  good hyperparameters
def train_and_predict(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net =getNet()
    train_l, _ = train(net,train_features, train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_l], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_l[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(-1))
    submission = pd.concat((test_data['Id'],test_data['SalePrice']),axis=1)
    submission.to_csv('mission.csv',index= False)


train_and_predict(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
d2l.plt.show()

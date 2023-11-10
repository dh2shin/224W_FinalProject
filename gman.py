import torch
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import GMAN
from torch_geometric.nn.models import Node2Vec

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from load_data import DataLoader

def get_TE(num_his, num_pred, batch_size=1):
    TE = torch.from_numpy(np.load("temp_embeddings.npy"))
    return TE
    # return TE[torch.randint(dim=1,high=TE.shape[0],size=(1,)).item(),:,:]
    # return torch.randn((batch_size, num_his + num_pred,2))

def get_SE(number_of_nodes,K,d):
    return torch.from_numpy(np.load("embeddings.npy"))
    # print("a size = ", a.shape)
    # breakpoint()
    # return torch.randn((325,64))
    # return torch.randn((number_of_nodes,K*d))
    # return a
    
def predict(model, test_dataloader, num_nodes=325):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # if batch_idx >= 10:
            #     break
            batch_x_adj = batch.x[:,0,:].squeeze().permute(1,0).unsqueeze(0)
            
            TE = get_TE(model.num_his, model.num_pred)
            SE = get_SE(num_nodes, model.K, model.d)
            
            output = model(batch_x_adj, SE, TE)
            
            true.append(batch.y[:,0,:].squeeze().permute(1,0).unsqueeze(0)[:,1,:].numpy())
            pred.append(output[:,0,:].numpy())
            # print("pred shape = ", output[:,0,:].numpy().shape)
            # print("true shape = ", batch.y[:,0,:].squeeze().permute(1,0).unsqueeze(0)[:,1,:].numpy().shape)
    # model.train()
    # Return in (timestep x num_features, num_nodes) dimension
    return np.stack(pred, axis=0).reshape((-1, num_nodes)), np.stack(true, axis=0).reshape((-1, num_nodes))
    
class GMAN_Traffic(torch.nn.Module):
    def __init__(self, L, K, d, num_his, bn_decay, steps_per_day, use_bias, mask):
        super(GMAN_Traffic, self).__init__()
        self.K = K
        self.d = d
        self.gman = GMAN(L, K, d, num_his, bn_decay, steps_per_day, use_bias, mask)
        self.num_pred = num_his
        self.num_his = num_his
        # self.linear = torch.nn.Linear(325, output_dim)
    
    def forward(self, X, SE, TE):
        # print("input dims = ", X.shape)
        h =  self.gman(X,SE,TE)
        # print("output dims = ", h.shape)
        return h
        

def mae_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
        
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    
if __name__ == "__main__": 
    '''
    L = number of STAtt blocks  
    K = number of attention heads
    d = dimensions of each head attention output
    num_his = number of history steps
    bn_decay = batch normalization decay
    steps_per_day = number of time steps per day
    use_bias = whether to use bias in FC
    mask = whether to mask attention score in temporal attention
    '''   
    model = GMAN_Traffic(L=1, K=8, d=8, num_his=12, bn_decay=0.99, steps_per_day=6, use_bias=True, mask=True)
    learning_rate = 0.001
    max_epoch = 10
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    pems_loader = DataLoader(baseline=False)
    pems_loader.load_data()
    
    
    train_loader = pems_loader.train
    test_loader = pems_loader.test
    
    optimizer.zero_grad()
    model.train()
    
    print("Loaded Data")
    # print(train_loader)
    for epoch in range(max_epoch):
        model.train()
        loss = 0
        step = 0
        for batch_index, batch_data in enumerate(train_loader):
            if batch_index > 10:
                break
            num_nodes = batch_data.x.shape[0]
            # batch_x_adj = torch.randn(325,12,325)
            batch_x_adj = batch_data.x[:,0,:].squeeze().permute(1,0).unsqueeze(0)
            # breakpoint()
            
            TE = get_TE(model.num_his, model.num_pred)
            SE = get_SE(num_nodes, model.K, model.d)
            # print("TE shape = " , TE.shape)
            # print("SE shape = ", SE.shape)
            # print("batch_x shape = ", batch_x_adj.shape)
            # breakpoint()
            output = model(batch_x_adj, SE, TE)
            # breakpoint()
            loss = loss + mae_loss(output, batch_data.y[:,0,:].squeeze().permute(1,0).unsqueeze(0))
            print("Step {:05d} | Avg. Loss {:.4f}".format(batch_index, loss.item()/(batch_index+1)))
        torch.save(model.state_dict(),"models/GMAN_Traffic_epoch{}.pt".format(epoch))
        loss = loss / (batch_index + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))
        
        model.eval()
        val_loss = 0
        step = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x_adj = batch_data.x[:,0,:].squeeze().permute(1,0).unsqueeze(0)
                TE = get_TE(model.num_his, model.num_pred)
                SE = get_SE(num_nodes, model.K, model.d)
                output = model(batch_x_adj, SE, TE)
                val_loss = val_loss + mae_loss(output, batch.y[:,0,:].squeeze().permute(1,0).unsqueeze(0))
                step += 1
            val_loss = val_loss / step
            print("Epoch {} validation MSE: {:.4f}".format(epoch, val_loss.item()))
    # print(model)
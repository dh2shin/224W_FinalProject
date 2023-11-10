import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import AGCRN
import pandas as pd
import numpy as np
from load_data import DataLoader
from evaluate_model import ModelEvaluator

class AGCRNWrapper(nn.Module):
    def __init__(self, epochs=10, learning_rate=0.01, embed_dim=64):
        super(AGCRNWrapper, self).__init__()
        data_loader = DataLoader(baseline=False)
        data_loader.load_data()
        self.train_dataloader = data_loader.train
        self.test_dataloader = data_loader.test
        self.num_nodes, self.feature_dim, self.num_features = self.train_dataloader.features[0].shape
        self.embedding_dim = embed_dim
        self.model = AGCRN(number_of_nodes=self.num_nodes, # 325
                           in_channels=self.num_features, # 12
                           out_channels=self.num_features, # 12
                           K=3,
                           embedding_dimensions=self.embedding_dim) # 64
        self.node_embeddings = nn.Parameter(torch.randn(self.feature_dim, self.embedding_dim), requires_grad=True) # 2 x 64
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
    
    def predict(self, mode):
        self.model.eval()
        pred = []
        true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                output = self.model(batch.x, self.node_embeddings)
                true.append(batch.y[:,0,:].numpy())
                pred.append(output[:,0,:].numpy())
        self.model.train()
        # Return in (timestep x num_features, num_nodes) dimension
        return np.stack(pred, axis=0).reshape((-1, self.num_nodes)), np.stack(true, axis=0).reshape((-1, self.num_nodes))

if __name__ == "__main__":
    model = AGCRNWrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    optimizer.zero_grad()
    model.train()
    print("Running training...")
    for epoch in range(15): 
        loss = 0
        step = 0
        for time, batch in enumerate(model.train_dataloader):
            out = model.model(batch.x, model.node_embeddings)
            loss = loss + model.loss_fn(out[:,0,:], batch.y[:,0,:])
            step += 1
        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
        torch.save(model.state_dict(), "PATH") # TODO define save path
        model.eval()
        val_loss = 0
        step = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(model.test_dataloader):
                output = model.model(batch.x, model.node_embeddings)
                val_loss = val_loss + model.loss_fn(output[:,0,:], batch.y[:,0,:])
                step += 1
            val_loss = val_loss / step
            print("Epoch {} validation MSE: {:.4f}".format(epoch, val_loss.item()))
        model.train()

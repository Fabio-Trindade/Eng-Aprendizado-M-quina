from torch import nn
import torch

class SpaceShipModel(nn.Module):
    def __init__(self,dim_total_data,dim_total_vocab_data,len_vocab,dim_embedd,dropout = 0.5):
        super(SpaceShipModel,self).__init__()
        self.embedding = nn.Embedding(len_vocab,dim_embedd,dtype=float)
        self.linear = nn.Linear((dim_total_data-dim_total_vocab_data) + dim_total_vocab_data*dim_embedd,dim_embedd,dtype=float)
        self.dim_total_vocab_data = dim_total_vocab_data
        self.dim_embedd = dim_embedd
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, features_embedding, remain_features,batch_size):
        x = self.embedding(features_embedding)
        x = x.reshape((batch_size,self.dim_total_vocab_data*self.dim_embedd))
        x = self.dropout(x)
        return  self.leaky_relu(self.linear(torch.concat([x,remain_features],dim=1)))


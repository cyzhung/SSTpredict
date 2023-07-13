#Model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import ERA5_Dataset

# class net(nn.Module):
#     def __init__(self,in_dim,embedding_dim=100,hidden_dim=1024):
#         super().__init__()

#         NUM_MONTH=12

#         self.monthEmbedding = nn.Embedding(NUM_MONTH+1, embedding_dim)

#         self.hidden1 = nn.Linear(in_dim+embedding_dim,hidden_dim,dtype=torch.float64)
#         self.hidden2 = nn.Linear(hidden_dim,hidden_dim*2,dtype=torch.float64)

#         self.out = nn.Linear(hidden_dim*2,1,dtype=torch.float64)

#         self.relu = nn.ReLU()



#     def forward(self,x,month):
        

#         monthEmbedding = self.monthEmbedding(month).squeeze(1)

        
#         x = torch.cat((monthEmbedding,x),dim=1)


      
#         x = self.hidden1(x)     
#         x = self.relu(x)

#         x = self.hidden2(x)     
#         x = self.relu(x)
#         x = self.out(x)

#         return x

class net(nn.Module):
    def __init__(self,in_dim,embedding_dim=50,hidden_dim=1024,Bidirection=False,num_layers=2):
        super().__init__()

        NUM_MONTH=12

        self.monthEmbedding = nn.Embedding(NUM_MONTH+1, embedding_dim)

        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=in_dim+embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.Bidirection = 2 if Bidirection==True else 1
        self.out = nn.Linear(hidden_dim*self.Bidirection,1)

        self.relu = nn.ReLU()

        self.hidden_dim = hidden_dim


    def forward(self,x):
        h0 = torch.zeros(self.num_layers * self.Bidirection, x.size(0), self.hidden_dim).to(next(self.parameters()).device,dtype=torch.float32)
        c0 = torch.zeros(self.num_layers * self.Bidirection, x.size(0), self.hidden_dim).to(next(self.parameters()).device,dtype=torch.float32)

        month = x[:,:,5]
        month = month.to(dtype=torch.int)

        input = x[:,:,:5]


        monthEmbedding = self.monthEmbedding(month).squeeze(1)
 
        input = torch.cat((monthEmbedding,input), dim=2)


        out, _ = self.rnn(input,(h0,c0))

        out = self.out(out[:,-1,:])


        return out


if __name__ =="__main__":
    testDataset = ERA5_Dataset(2021,num_previousMonth=6)
    testLoader = DataLoader(dataset=testDataset,batch_size=8,shuffle=True)

    model = net(5).to("cuda")
    for data in testLoader:
        feature = data['feature'].to("cuda")
        targetSST = data['targetSST'].to("cuda")
        out = model(feature)
        print(out)

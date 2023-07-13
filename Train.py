#Train.py

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader,ConcatDataset
from Dataset import ERA5_Dataset
from Model import net
from test import test
device = "cuda" if torch.cuda.is_available() else "cpu"

# def train(model,trainLoader,testLoader,lr,epochs,loss_fn=nn.MSELoss()):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         with tqdm(total=len(trainLoader),desc=f'Epoch:{epoch}') as progress:
#             trainLoss = 0
#             for data in trainLoader:
#                 sst = data['sst'].to(device).unsqueeze(1)
#                 tp = data['tp'].to(device).unsqueeze(1)
#                 t2m = data['t2m'].to(device).unsqueeze(1)
                
#                 lat = data['lat'].to(device).unsqueeze(1)
#                 lon = data['lon'].to(device).unsqueeze(1)
#                 month = data['month'].to(device).unsqueeze(1)
#                 lastMonthSST = data['lastSST'].to(device).unsqueeze(1)

#                 features = torch.concat([tp,t2m,lat,lon,lastMonthSST],dim=-1)
                

#                 out = model(features,month).to(device)
                
#                 loss = loss_fn(out,sst).to(torch.float32)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 progress.set_description(f'Epoch:{epoch},Loss:{loss.item()}')
#                 progress.update()
#                 trainLoss += loss.item()
                
#             trainLoss /= len(trainLoader)

#             testLoss = test(model,testLoader)
#             progress.set_description(f'Epoch:{epoch},TrainLoss:{trainLoss},TestLoss:{testLoss}')

#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
                
#                 }, f"net_{epoch}_v4.pt")
            

# def train(model,trainLoader,testLoader,lr,epochs,loss_fn=nn.MSELoss()):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         with tqdm(total=len(trainLoader),desc=f'Epoch:{epoch}') as progress:
#             trainLoss = 0
#             for data in trainLoader:
#                 sst = data['sst'].to(device).unsqueeze(1)
#                 tp = data['tp'].to(device).unsqueeze(1)
#                 t2m = data['t2m'].to(device).unsqueeze(1)
                
#                 lat = data['lat'].to(device).unsqueeze(1)
#                 lon = data['lon'].to(device).unsqueeze(1)
#                 month = data['month'].to(device).unsqueeze(1)
#                 lastMonthSST = data['lastSST'].to(device).unsqueeze(1)

#                 features = torch.concat([tp,t2m,lat,lon,lastMonthSST],dim=-1)
                

#                 out = model(features,month).to(device)
                
#                 loss = loss_fn(out,sst).to(torch.float32)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 progress.set_description(f'Epoch:{epoch},Loss:{loss.item()}')
#                 progress.update()
#                 trainLoss += loss.item()
                
#             trainLoss /= len(trainLoader)

#             testLoss = test(model,testLoader)
#             progress.set_description(f'Epoch:{epoch},TrainLoss:{trainLoss},TestLoss:{testLoss}')

#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
                
#                 }, f"net_{epoch}_v4.pt")

# def test(model,testLoader,loss_fn=nn.MSELoss()):
#     model.eval()

#     testLoss = 0

#     months = []
#     predicts = []
#     labels = []
#     latitudes = []
#     longitudes = []
#     lastSST = []
#     with tqdm(total=len(testLoader)) as progress:
#         for data in testLoader:
#             sst = data['sst'].to(device).unsqueeze(1)
#             tp = data['tp'].to(device).unsqueeze(1)
#             t2m = data['t2m'].to(device).unsqueeze(1)
            
#             lat = data['lat'].to(device).unsqueeze(1)
#             lon = data['lon'].to(device).unsqueeze(1)
#             month = data['month'].to(device).unsqueeze(1)
#             lastMonthSST = data['lastSST'].to(device).unsqueeze(1)


#             features = torch.concat([tp,t2m,lat,lon,lastMonthSST],dim=-1)

#             out = model(features,month).to(device)
            
#             loss = loss_fn(out,sst).to(torch.float32)

#             testLoss += loss.item()

#             latitudes += lat.tolist()
#             longitudes += lon.tolist()
#             months += (month.tolist())
#             predicts += (out.tolist())
#             labels += (sst.tolist())
#             lastSST += (lastMonthSST.tolist())
#             progress.update()


#     months = [int(element[0]) for element in months]
#     latitudes = [float(element[0]) for element in latitudes]
#     predicts = [float(element[0]) for element in predicts]
#     labels = [float(element[0]) for element in labels]
#     longitudes = [float(element[0]) for element in longitudes]
#     lastSST = [float(element[0]) for element in lastSST]


#     df = pd.DataFrame({"月份":months,"經度":latitudes,"緯度":longitudes,"海平面溫度預測結果(°C)":predicts,"海平面溫度實際結果(°C)":labels,"上個月海平面溫度":lastSST})

#     df["預測與實際結果歐式距離(°C)"] = (((df["海平面溫度實際結果(°C)"]- df["海平面溫度預測結果(°C)"])**2)**0.5)

#     df.to_csv("SST_Result_v3.csv")

#     return testLoss/len(testLoader)

def train(model,trainLoader,testLoader,lr,epochs,loss_fn=nn.MSELoss()):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        modelName = f"RNNnet_{epoch}"
        with tqdm(total=len(trainLoader),desc=f'Epoch:{epoch}') as progress:
            trainLoss = 0
            for data in trainLoader:
                features = data['feature'].to(device,dtype=torch.float32)
                targetSST = data['targetSST'].to(device,dtype=torch.float32).unsqueeze(1)
                out = model(features).to(device,dtype=torch.float32)
                
                loss = loss_fn(out,targetSST).to(torch.float32)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_description(f'Epoch:{epoch},Loss:{loss.item()}')
                progress.update()
                trainLoss += loss.item()
                
            trainLoss /= len(trainLoader)

            testLoss = test(modelName,model,testLoader)
            progress.set_description(f'Epoch:{epoch},TrainLoss:{trainLoss},TestLoss:{testLoss}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                
                }, modelName+".pt")

if __name__=="__main__":

    startYear = 2015
    endYear = 2020

    model = net(in_dim=5).to(device)


    # model.load_state_dict(checkpoint["model_state_dict"])

    num_month = 12
    trainDataset = ERA5_Dataset(startYear,num_previousMonth=num_month)
    for year in range(startYear+1,endYear+1):
        trainDataset = ConcatDataset([trainDataset,ERA5_Dataset(year,num_previousMonth=num_month)])
    trainLoader = DataLoader(dataset=trainDataset,batch_size=32,shuffle=True)


    testDataset = ERA5_Dataset(2021,num_previousMonth=num_month)
    testLoader = DataLoader(dataset=testDataset,batch_size=32)

    

    train(model,trainLoader,testLoader,lr=0.001,epochs=1)

    



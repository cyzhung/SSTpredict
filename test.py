#test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
import pandas as pd
from tqdm import tqdm
from Dataset import ERA5_Dataset
from Model import net

device = "cuda"

def test(modelName,model,testLoader,loss_fn=nn.MSELoss()):
    model.eval()

    testLoss = 0

    months = []
    predicts = []
    labels = []
    latitudes = []
    longitudes = []
    lastSST = []
    with tqdm(total=len(testLoader)) as progress:
        for data in testLoader:
            features = data['feature'].to(device)
            targetSST = data['targetSST'].to(device).unsqueeze(1)

            out = model(features).to(device)
            
            loss = loss_fn(out,targetSST)

            testLoss += loss.item()

            lat = features[:,-1,3]
            lon = features[:,-1,4]
            month = features[:,-1,5]
            lastMonthSST = features[:,0,1]
            latitudes += lat.tolist()
            longitudes += lon.tolist()

            months += (month.to(dtype=torch.int).tolist())
            predicts += (out.tolist())


            labels += (targetSST.tolist())
            lastSST += (lastMonthSST.tolist())

            progress.update()



    labels = [float(element[0]) for element in labels]
    predicts = [float(element[0]) for element in predicts]


    df = pd.DataFrame({"月份":months,"經度":latitudes,"緯度":longitudes,"海平面溫度預測結果(°C)":predicts,"海平面溫度實際結果(°C)":labels,"上個月海平面溫度":lastSST})

    df["預測與實際結果歐式距離(°C)"] = (((df["海平面溫度實際結果(°C)"] - df["海平面溫度預測結果(°C)"])**2)**0.5)

    df.to_csv(modelName+".csv")
    computeAverageError(modelName+".csv")
    return testLoss/len(testLoader)

def computeAverageError(path):
    df = pd.read_csv(path)
    print(f"PathName:{path}")
    print("預測與實際結果歐式距離(°C):",(df["預測與實際結果歐式距離(°C)"]).sum() / len(df))

    return (df["預測與實際結果歐式距離(°C)"]).sum() / len(df)
if __name__=="__main__":

    model = net(in_dim=5).to(device)

    modelName = "RNNnet_0_m9"
    checkpoint = torch.load(modelName+".pt")
    model.load_state_dict(checkpoint["model_state_dict"],False)


    testDataset = ERA5_Dataset(2021,num_previousMonth=12)


    testLoader = DataLoader(dataset=testDataset,batch_size=1)
    test(modelName,model,testLoader)



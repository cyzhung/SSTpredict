#Dataset.py


import numpy as np
from datetime import datetime
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader,ConcatDataset


# class ERA5_Dataset(Dataset):
#     def __init__(self,year):
#         self.csv1 = pd.read_csv(f"Merge_{year}.csv")
#         self.csv2 = pd.read_csv(f"Merge_{year+1}.csv")
#         self.year = year
 #        self.num_EveryMonthData = 686364
#     def __getitem__(self,idx):
        
#         time = self.csv1['time'][idx]
#         time = datetime.fromisoformat(time)
#         t2m = self.csv1['t2m'][idx]
#         lon = self.csv1['longitude'][idx]
#         lat = self.csv1['latitude'][idx]
#         tp = self.csv1['tp'][idx]
#         lastMonthSST = self.csv1['sst'][idx]
#         month = int(idx / self.num_EveryMonthData) + 1

#         if month == 12:
#             idx = idx % self.num_EveryMonthData
#             df = self.csv2
#         else:
#             idx += self.num_EveryMonthData
#             df = self.csv1
        
       

#         sst = df['sst'][idx].item()
#         # print(df.iloc[idx]['longitude']  ,"  ", lon)
#         # print(df.iloc[idx]['latitude']  ,"  ", lat)
#         # print(df.iloc[idx]['time']  ,"  ", time.year,time.month)

#         data = {'tp':tp,'sst':sst,'t2m':t2m,'lon':lon,'lat':lat,'month':time.month,'lastSST':lastMonthSST}

#         return data

#     def __len__(self):
#         return len(self.csv1)

class ERA5_Dataset(Dataset):
    def __init__(self,year,num_previousMonth=3):
        self.lastYearDF = pd.read_csv(f"Merge_{year}.csv")
        self.currentYearDF = pd.read_csv(f"Merge_{year+1}.csv")
        self.year = year
        self.num_EveryMonthData = 686364
        self.num_previousMonth = num_previousMonth
    def findSomeMonthData(self,targetMonth,currentMonth,idx):
        if(targetMonth >= currentMonth):
            monthDifference = targetMonth - currentMonth
            idx = idx + self.num_EveryMonthData*monthDifference
            df = self.lastYearDF
        else:
            monthDifference = currentMonth - targetMonth
            idx = idx - self.num_EveryMonthData*monthDifference
            df = self.currentYearDF


        tp = df['tp'][idx]
        sst = df['sst'][idx]
        t2m = df['t2m'][idx]
        lon = df['longitude'][idx]
        lat = df['latitude'][idx]
        time = df['time'][idx]
        time = datetime.fromisoformat(time)

        return {'tp':tp,'sst':sst,'t2m':t2m,'lon':lon,'lat':lat,'month':time.month}        

    def __getitem__(self,idx):
        
        time = self.currentYearDF['time'][idx]
        targetSST = self.currentYearDF['sst'][idx]
        time = datetime.fromisoformat(time)
        


        
        MONTH = [1,2,3,4,5,6,7,8,9,10,11,12]




        feature = []
        for i in range(1,self.num_previousMonth+1):
            targetMonthidx = time.month - i - 1
            targetMonth = MONTH[targetMonthidx]
            data = self.findSomeMonthData(targetMonth,time.month,idx)

            feature.append([data['tp'],data['sst'],data['t2m'],data['lon'],data['lat'],data['month']])

        feature = torch.tensor(feature).to(dtype=torch.float32)


        return {'feature':feature,'targetSST':targetSST}

    def __len__(self):
        return len(self.currentYearDF)
    
if __name__=="__main__":
    testDataset = ERA5_Dataset(2021)
    testLoader = DataLoader(dataset=testDataset,batch_size=1,shuffle=True)

    for data in testLoader:
        feature = data['feature']
        targetSST = data['targetSST']
        print(feature.shape)
        print(feature)
        print(targetSST)
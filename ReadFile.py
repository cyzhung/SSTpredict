#ReadFile.py

import os
import numpy as np
import netCDF4 as nc
import pandas as pd
from tqdm import tqdm
def to_csv(file,variable):

    
    # Extract variable
    t2m = file.variables[variable]
    
    # Get dimensions assuming 3D: time, latitude, longitude
    time_dim, lat_dim, lon_dim = t2m.get_dims()
    time_var = file.variables[time_dim.name]
    times = nc.num2date(time_var[:], time_var.units)
    latitudes = file.variables[lat_dim.name][:]
    longitudes = file.variables[lon_dim.name][:]
    
    output_dir = './'
    
    # =============================== METHOD 2 ============================
    # Write data as a table with 4 columns: time, latitude, longitude, value
    # =====================================================================
    filename = os.path.join(output_dir, f'{variable}_{year}.csv')
    print(f'Writing data in tabular form to {filename} (this may take some time)...')
    times_grid, latitudes_grid, longitudes_grid = [
        x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]
    df = pd.DataFrame({
        'time': [t.isoformat() for t in times_grid],
        'latitude': latitudes_grid,
        'longitude': longitudes_grid,
         variable : t2m[:].flatten()})
    df.to_csv(filename, index=False)
    print('Done')


if __name__ =="__main__":


    # create CSV
    # startYear = 2021
    # endYear = 2022
    # for year in range(startYear,endYear+1):
    # year = 2021
    # preciFile = nc.Dataset(f'ERA5/preci_monthly_average/preci_{year}_monthly_average.nc')
    # sstFile = nc.Dataset(f'ERA5/sst_monthly_average/sst_{year}_monthly_average.nc')
    # t2mFile = nc.Dataset(f'ERA5/t2m_monthly_average/t2m_{year}_monthly_average.nc')


    # to_csv(preciFile,"tp")
    # to_csv(sstFile,"sst")
    # to_csv(t2mFile,"t2m")
    # preciFile.close()
    # sstFile.close()
    # t2mFile.close()

    ##concate csv

    # startYear = 2010
    # endYear = 2022

    # with tqdm(total=endYear-startYear+1) as pro:
    #     for year in range(startYear,endYear+1):
    #         sst = pd.read_csv(f"sst_{year}.csv")
    #         t2m = pd.read_csv(f"t2m_{year}.csv")
    #         tp = pd.read_csv(f"tp_{year}.csv")


    #         mergeData = pd.concat([sst,t2m.loc[:,"t2m"],tp.loc[:,'tp']],axis=1)
    #         mergeData.dropna(inplace=True)

    #         mergeData.to_csv(f"Merge_{year}.csv")

    #         pro.update()

    df = pd.read_csv("RNNnet_0_m9.csv")
    print((df["預測與實際結果歐式距離(°C)"]).sum() / len(df))
    print(abs(df["上個月海平面溫度"]-df["海平面溫度實際結果(°C)"]).sum() / len(df))




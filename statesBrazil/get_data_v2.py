import pandas as pd
import urllib.request
import numpy as np

def datadownload():
    print('Baixando arquivos brasil.io...')
    # fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
    url = 'https://data.brasil.io/dataset/covid19/caso.csv.gz'
        
    class AppURLopener(urllib.request.FancyURLopener):
        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
    
    urllib._urlopener = AppURLopener()
    urllib._urlopener.retrieve(url, "data/BR.csv.gz")

def group():
    dataBR = pd.read_csv("data/dados_total_estados.csv", index_col=[0],compression='gzip')
    state = dataBR["state"].unique()
    df_confirmed = dataBR.groupby(['date','state'],as_index = False).sum().pivot('date','state').fillna(0)['confirmed']
    df_deaths = dataBR.groupby(['date','state'],as_index = False).sum().pivot('date','state').fillna(0)['deaths']
    df_popEst = dataBR.groupby(['date','state'],as_index = False).sum().pivot('date','state').fillna(0)['popEst']
    
    df_popEst= df_popEst.replace(0, np.nan)
    df_popEst = df_popEst.reindex(pd.date_range(df_popEst.index.min(), df_popEst.index.max()), fill_value=np.nan)
    df_popEst = df_popEst.interpolate(method='akima', axis=0).ffill().bfill()
    
    df_popEst = df_popEst[df_popEst.iloc[:,0]>0]
    df_popEst =  df_popEst.iloc[-1,:]
    df_popEst =  df_popEst.reset_index()
    df_popEst.columns = ['state', 'popEst']
    df_popEst = df_popEst.replace(['Indefinido'],'Brazil')
    print(df_popEst)
    
    df_confirmed.to_csv("data/confirmados.csv", sep=",")
    df_deaths.to_csv("data/mortes.csv", sep=",")
    df_popEst.to_csv("data/popEst.csv", sep=",")

def get_data():
    datadownload()
    dataBR = pd.read_csv('data/BR.csv.gz', compression='gzip')
    dataBR = dataBR.rename(columns={'dateRep': 'date', 'estimated_population_2019': 'popEst'})

    for i in range(len(dataBR)):
            if dataBR["city"][i] != dataBR["city"][i]:
                dataBR.at[i, "city"] = "TOTAL" 

    df = dataBR.query('city == "TOTAL"').reset_index()
    df.to_csv("data/dados_total_estados.csv", sep=",", index=False,compression='gzip')

    group()
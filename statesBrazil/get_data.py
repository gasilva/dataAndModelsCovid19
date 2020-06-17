import pandas as pd

from urllib.request import Request, urlopen

def datadownload():
    print('Baixando arquivos brasil.io...')
    # fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
    url = 'https://data.brasil.io/dataset/covid19/caso.csv.gz'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

def group():
    dataBR = pd.read_csv("data/dados_total_estados.csv", index_col=[0])
    state = dataBR["state"].unique()
    df_confirmed = dataBR.groupby(['date','state'],as_index = False).sum().pivot('date','state').fillna(0)['confirmed']
    df_deaths = dataBR.groupby(['date','state'],as_index = False).sum().pivot('date','state').fillna(0)['deaths']
    
    df_confirmed.to_csv("data/confirmados.csv", sep=",")
    df_deaths.to_csv("data/mortes.csv", sep=",")

def get_data():
    datadownload()
    dataBR = pd.read_csv('data/BR.csv.gz', compression='gzip')
    dataBR = dataBR.rename(columns={'dateRep': 'date', 'estimated_population_2019': 'popEst'})

    for i in range(len(dataBR)):
            if dataBR["city"][i] != dataBR["city"][i]:
                dataBR.at[i, "city"] = "TOTAL" 

    df = dataBR.query('city == "TOTAL"').reset_index()
    df.to_csv("data/dados_total_estados.csv", sep=",", index=False)

    group()
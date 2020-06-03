import urllib.request
import pandas as pd

def datadownload():
    print('Baixando arquivos brasil.io...')
    # fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
    url = 'https://data.brasil.io/dataset/covid19/caso.csv.gz'
    urllib.request.urlretrieve(url, 'data/BR.csv.gz')

def get_data():
    datadownload()
    dataBR = pd.read_csv('data/BR.csv.gz', compression='gzip')
    dataBR = dataBR.rename(columns={'dateRep': 'date', 'estimated_population_2019': 'popEst'})

    for i in range(len(dataBR)):
            if dataBR["city"][i] != dataBR["city"][i]:
                dataBR.at[i, "city"] = "TOTAL" 

    df = dataBR.query('city == "TOTAL"').reset_index()
    df.to_csv("data/dados_total_estados.csv", sep=",", index=False)
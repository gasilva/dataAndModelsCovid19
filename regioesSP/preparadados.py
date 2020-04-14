import urllib.request
import pandas as pd

def baixadados():
    print('Baixando arquivos brasil.io...')
    # fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
    url = 'https://data.brasil.io/dataset/covid19/caso.csv.gz'
    urllib.request.urlretrieve(url, 'data/BR.csv.gz')
    
def organiza_regiao_SP():
    dadosBR = pd.read_csv('data/dados_municipios_Brasil.csv', index_col=[0])
    dfSP = dadosBR.query('state == "SP"').reset_index()
    infoMunicipios = pd.read_csv('data/Municipios_para_Predicao_Covid19.csv')
    
    drs = []
    for i in range(len(dfSP)):
        try:
            index = list(infoMunicipios["mun_descr"]).index(dfSP["city"][i])
            drs.append(infoMunicipios["dir_descr"][index])
        except:
            drs.append("Indefinido")
    dfSP["DRS"] = drs
    DRS = dfSP["DRS"].unique()
    df_confirmed = dfSP.groupby(['date','DRS'],as_index = False).sum().pivot('date','DRS').fillna(0)['confirmed']
    df_deaths = dfSP.groupby(['date','DRS'],as_index = False).sum().pivot('date','DRS').fillna(0)['deaths']
    
    dfSP.to_csv("data/dados_municipios_SP.csv", sep=",", index=False)
    df_confirmed.to_csv("data/DRS_confirmados.csv", sep=",")
    df_deaths.to_csv("data/DRS_mortes.csv", sep=",")

def preparadados():
    baixadados()
    dadosBR = pd.read_csv('data/BR.csv.gz', compression='gzip')
    dadosBR = dadosBR.rename(columns={'dateRep': 'date', 'estimated_population_2019': 'popEst'})
    for i in range(len(dadosBR)):
        # Corrige NaN total estado
        if dadosBR["city"][i] != dadosBR["city"][i]:
            dadosBR.at[i, "city"] = "TOTAL"    
    dadosBR.to_csv("data/dados_municipios_Brasil.csv", sep=",", index=False)
    
    organiza_regiao_SP()   

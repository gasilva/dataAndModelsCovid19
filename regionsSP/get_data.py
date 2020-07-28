import pandas as pd
import urllib.request

def datadownload():
    print('Baixando arquivos brasil.io...')
    # fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
    url = 'https://data.brasil.io/dataset/covid19/caso.csv.gz'
        
    class AppURLopener(urllib.request.FancyURLopener):
        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
    
    urllib._urlopener = AppURLopener()
    urllib._urlopener.retrieve(url, "data/BR.csv.gz")
    
def group_by_DRS():
    dataBR = pd.read_csv('data/dados_municipios_Brasil.csv', index_col=[0])
    dfSP = dataBR.query('state == "SP"').reset_index()
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
    df_popEst = dfSP.groupby(['date','DRS'],as_index = False).sum().pivot('date','DRS').fillna(0)['popEst']
    df_popEst = df_popEst.tail(1)
    df_popEst =  df_popEst.reset_index()
    df_popEst = df_popEst.drop('date', 1)
    df_popEst = df_popEst.T
    df_popEst =  df_popEst.reset_index()
    df_popEst.columns = ['DRS', 'popEst']
    df_popEst = df_popEst.replace(['Indefinido'],'SP')

    dfSP.to_csv("data/dados_municipios_SP.csv", sep=",", index=False)
    df_confirmed.to_csv("data/DRS_confirmados.csv", sep=",")
    df_deaths.to_csv("data/DRS_mortes.csv", sep=",")
    df_popEst.to_csv("data/DRS_popEst.csv", sep=",")

def get_data():
    datadownload()
    dataBR = pd.read_csv('data/BR.csv.gz', compression='gzip')
    dataBR = dataBR.rename(columns={'dateRep': 'date', 'estimated_population_2019': 'popEst'})
    for i in range(len(dataBR)):
        if dataBR["city"][i] != dataBR["city"][i]:
            dataBR.at[i, "city"] = "TOTAL"    
    dataBR.to_csv("data/dados_municipios_Brasil.csv", sep=",", index=False)
    
    group_by_DRS()   
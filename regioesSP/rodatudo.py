import preparadados
import pandas as pd
import os

preparadados.preparadados() #baixa dados e organiza

dfSP = pd.read_csv("data/dados_municipios_SP.csv")
dfparam = pd.read_csv("data/param.csv")

DRS = list(dfSP["DRS"].unique())
DRS.remove("Indefinido")

for drs in DRS:
    comando = "python SEAIRD_DRS.py"
    query = dfparam.query('DRS == "{}"'.format(drs)).reset_index()
    try:
        comando += " --states '{}'".format(drs)
        if query['start-date'][0] == query['start-date'][0]:
            comando += " --start-date '{}'".format(query['start-date'][0])
        elif query['prediction-range'][0] == query['prediction-range'][0]:
            comando += " --prediction-days '{}'".format(query['prediction-range'][0])
        elif query['s0'][0] == query['s0'][0]:
            comando += " --S_0 '{}'".format(query['s0'][0])
        elif query['e0'][0] == query['e0'][0]:
            comando += " --E_0 '{}'".format(query['e0'][0])
        elif query['a0'][0] == query['a0'][0]:
            comando += " --A0 '{}'".format(query['a0'][0])
        elif query['i0'][0] == query['i0'][0]:
            comando += " --I0 '{}'".format(query['i0'][0])
        elif query['r0'][0] == query['r0'][0]:
            comando += " --R0 '{}'".format(query['r0'][0])
        elif query['d0'][0] == query['d0'][0]:
            comando += " --D0 '{}'".format(query['d0'][0])
        print(comando)  
        os.system(comando)
    except Exception as e: 
        print(e)

import get_data
import pandas as pd
import os

get_data.get_data() #baixa dados e organiza

dfSP = pd.read_csv("data/dados_municipios_SP.csv")
dfparam = pd.read_csv("data/param.csv")

DRS = list(dfSP["DRS"].unique())
DRS.remove("Indefinido")

for drs in DRS:
    command  = "python SEAIRD_DRS.py"
    query = dfparam.query('DRS == "{}"'.format(drs)).reset_index()
    try:
        command  +=' --districtRegions "{}"'.format(drs)
        if query['start-date'][0] == query['start-date'][0]:
            command  +=' --start-date {}'.format(query['start-date'][0])
        if query['prediction-range'][0] == query['prediction-range'][0]:
            command  +=' --prediction-days {}'.format(int(query['prediction-range'][0]))
        if query['s0'][0] == query['s0'][0]:
            command  +=' --S_0 {}'.format(query['s0'][0])
        if query['e0'][0] == query['e0'][0]:
            command  +=' --E_0 {}'.format(query['e0'][0])
        if query['a0'][0] == query['a0'][0]:
            command  +=' --A_0 {}'.format(query['a0'][0])
        if query['i0'][0] == query['i0'][0]:
            command  +=' --I_0 {}'.format(query['i0'][0])
        if query['r0'][0] == query['r0'][0]:
            command  +=' --R_0 {}'.format(query['r0'][0])
        if query['d0'][0] == query['d0'][0]:
            command  +=' --D_0 {}'.format(query['d0'][0])
        if query['START'][0] == query['START'][0]:
            command  +=' --START {}'.format(int(query['START'][0]))
        if query['RATIO'][0] == query['RATIO'][0]:
            command  +=' --RATIO {}'.format(query['RATIO'][0])
        if query['WCASES'][0] == query['WCASES'][0]:
            command  +=' --WCASES {}'.format(query['WCASES'][0])
        if query['WREC'][0] == query['WREC'][0]:
            command  +=' --WREC {}'.format(query['WREC'][0])
        print(command)  
        os.system(command )
    except Exception as e: 
        print(e)

import get_data
import pandas as pd
import os

#precisa atualizar essa lista de parâmetros
#e mudar o param.csv

get_data.get_data() #baixa dados e organiza

dfSP = pd.read_csv("data/dados_municipios_SP.csv")
dfparam = pd.read_csv("data/param.csv")

DRS = list(dfSP["DRS"].unique())
DRS.remove("Indefinido")

for drs in DRS:
    command  = "python SEAIRD_DRS.py"
    query = dfparam.query('DRS == "{}"'.format(drs)).reset_index()
    try:
        command  += ' --districtRegions "{}"'.format(drs)
        if query['start-date'][0] == query['start-date'][0]:
            command  += ' --start-date {}'.format(query['start-date'][0])
        elif query['prediction-range'][0] == query['prediction-range'][0]:
            command  += ' --prediction-days {}'.format(query['prediction-range'][0])
        elif query['s0'][0] == query['s0'][0]:
            command  += " --S_0 {}".format(query['s0'][0])
        elif query['e0'][0] == query['e0'][0]:
            command  += " --E_0 {}".format(query['e0'][0])
        elif query['a0'][0] == query['a0'][0]:
            command  += " --A0 {}".format(query['a0'][0])
        elif query['i0'][0] == query['i0'][0]:
            command  += " --I0 {}".format(query['i0'][0])
        elif query['r0'][0] == query['r0'][0]:
            command  += " --R0 {}".format(query['r0'][0])
        elif query['d0'][0] == query['d0'][0]:
            command  += " --D0 {}".format(query['d0'][0])
        elif query['START'][0] == query['START'][0]:
            command  += " --START {}".format(query['START'][0])
        elif query['RATIO'][0] == query['RATIO'][0]:
            command  += " --RATIO {}".format(query['RATIO'][0])
        elif query['WCASES'][0] == query['WCASES'][0]:
            command  += " --WCASES {}".format(query['WCASES'][0])
        elif query['WREC'][0] == query['WREC'][0]:
            command  += " --WREC {}".format(query['WREC'][0])
        print(command )  
        os.system(command )
    except Exception as e: 
        print(e)

import numpy as np
import pandas as pd

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
df=pd.read_csv('https://brasil.io/dataset/covid19/caso?format=csv', \
    delimiter=',',parse_dates=True, date_parser=dateparse,header=None)
df.columns = df.iloc[0]
df=df.drop(df.index[0])

# for col in df.columns: 
#     print(col) 

cities = ["São Paulo","Rio de Janeiro", "Manaus", "Fortaleza", "Belo Horizonte"]
df.set_index("date", inplace=True)

ddf1=df[df.city == cities[0]].sort_index()
ddf2=df[df.city == cities[1]].sort_index()
ddf3=df[df.city == cities[2]].sort_index()
ddf4=df[df.city == cities[3]].sort_index()
ddf5=df[df.city == cities[4]].sort_index()

df_finalCases = pd.DataFrame({cities[0] : ddf1.confirmed,
                            cities[1] : ddf2.confirmed,
                            cities[2] : ddf3.confirmed,
                            cities[3] : ddf4.confirmed,
                            cities[4] : ddf5.confirmed,
                    }, index=ddf1.index)

df_finalCases.to_csv('./data/casesBrazilCities.csv')

df_finalDeaths = pd.DataFrame({cities[0] : ddf1.deaths,
                            cities[1] : ddf2.deaths,
                            cities[2] : ddf3.deaths,
                            cities[3] : ddf4.deaths,
                            cities[4] : ddf5.deaths,
                    }, index=ddf1.index)

df_finalDeaths.to_csv('./data/deathsBrazilCities.csv')
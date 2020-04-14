import numpy as np
import pandas as pd

#import csv
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
df=pd.read_csv('https://brasil.io/dataset/covid19/caso?format=csv', \
    delimiter=',',parse_dates=True, date_parser=dateparse,header=None)

#first line as columns
df.columns = df.iloc[0]
#drop duplicate line and name columns
df=df.drop(df.index[0])

# list of the cities
cities = ["São Paulo","Rio de Janeiro", "Manaus", "Fortaleza", "Belo Horizonte","Vitória","Curitiba","Macapá"]
# set the index
df.set_index("date", inplace=True)

# for col in df.columns: 
#      print(col) 

df_finalCases = pd.DataFrame({cities[0] : df[df.city == cities[0]].confirmed},
                                index=df[df.city == cities[0]].index)
for i in range(1,len(cities)):
    df_temp=pd.DataFrame({cities[i]:df[df.city == cities[i]].confirmed},
                                index=df[df.city == cities[0]].index)
    df_finalCases=pd.concat([df_finalCases,df_temp],axis=1)

df_finalCases=df_finalCases.sort_index()
df_finalCases.to_csv('./data/casesBrazilCities.csv')
df_finalCases.to_pickle('./data/casesBrazilCities.pkl')

df_finalDeaths = pd.DataFrame({cities[0] : df[df.city == cities[0]].deaths},
                                index=df[df.city == cities[0]].index)
for i in range(1,len(cities)):
    df_temp=pd.DataFrame({cities[i]:df[df.city == cities[i]].deaths},
                                index=df[df.city == cities[0]].index)
    df_finalDeaths=pd.concat([df_finalDeaths,df_temp],axis=1)

df_finalDeaths=df_finalDeaths.sort_index()
df_finalDeaths.to_csv('./data/deathsBrazilCities.csv')
df_finalDeaths.to_pickle('./data/deathsBrazilCities.pkl')
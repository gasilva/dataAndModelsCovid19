import matplotlib
matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
# sns.set()
import matplotlib.style as style
style.use('fivethirtyeight')

countries=["Italy","France","Brazil"]
version=420

for country in countries:
    versionStr=str(version)
    histOptAll= pd.read_table('./results/history_'+country+versionStr+'.csv', sep=",", index_col=0, header=None, 
        names = ["iterations","country","gtot",\
            "s0","startdate","i0","wcases","wrec",\
            "beta","beta2","sigma","sigma2","sigma3","gamma","b","mu"])
    histOptAll=histOptAll.dropna()
    histOptAll = histOptAll.reset_index(drop=True)

    histOpt=histOptAll[histOptAll.gtot==min(histOptAll.gtot)]
    print("---------------------------------------------------------------")
    print("fitting initial conditions")
    print(f"s0={int(histOpt.iloc[0]['s0']+0.5)}, date={int(histOpt.iloc[0]['startdate']+0.5)}, i0={int(histOpt.iloc[0]['i0']+0.5)}, wrec={histOpt.iloc[0]['wrec']:.4f}, wcases={histOpt.iloc[0]['wcases']:.4f}")
    print("parameters")
    print(f"beta={histOpt.iloc[0]['beta']:.8f}, beta2={histOpt.iloc[0]['beta2']:.8f}, 1/sigma={1/float(histOpt.iloc[0]['sigma']):.1f}")
    print(f"1/sigma2={1/float(histOpt.iloc[0]['sigma2']):.1f},1/sigma3={1/float(histOpt.iloc[0]['sigma3']):.1f}, gamma={histOpt.iloc[0]['gamma']:.8f}, b={histOpt.iloc[0]['b']:.8f}")
    print(f"mu={histOpt.iloc[0]['mu']:.8f}, r_0:{(histOpt.iloc[0]['beta']/histOpt.iloc[0]['gamma'])}")
    print("objective function")
    print("f(x)={}".format(histOpt.iloc[0]['gtot'])+" at iteration {}".format(histOpt.index.values[0]))
    print(country+" is done!")

    fig, ax = plt.subplots()
    fte_graph=histOptAll.plot(kind='line',y='gtot',figsize = (12,8))

    x=histOptAll.index[histOptAll.gtot<=(np.mean(histOptAll.gtot)-.5*np.std(histOptAll.gtot))]
    y=histOptAll.gtot[histOptAll.gtot<=(np.mean(histOptAll.gtot)-.5*np.std(histOptAll.gtot))]
    plt.plot(x, y,label="< ($\mu - 0.5 \cdot \sigma$)")

    histMin=histOptAll.nsmallest(5, ['gtot'])
    plt.scatter(histMin.index, histMin.gtot,label="5 lowest",c='green',marker='*',s=200)
    histOptAll.rolling(100).mean()['gtot'].plot(label="100th average")

    # Adding a title and a subtitle
    fte_graph.text(x = 0.15, y = 1.85, s = "Initial Conditions Optimization - "+country,
                fontsize = 26, weight = 'bold', alpha = .75,transform=ax.transAxes)
    fte_graph.text(x = 0.15, y = 1.77,
                s = "evolutionary optimization by Yabox",
                fontsize = 19, alpha = .85,transform=ax.transAxes)

    plt.legend()
    plt.savefig('./results/convergence_'+country+versionStr+'.png')
    plt.clf()

    version+=1

import matplotlib
matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import matplotlib.style as style
style.use('fivethirtyeight')
import dataFit_SEAIRD_v3AdjustIC as seaird
import matplotlib.font_manager as fm
# Font Imports
heading_font = fm.FontProperties(fname='/home/ats4i/playfair-display/PlayfairDisplay-Regular.ttf', size=24)
subtitle_font = fm.FontProperties(fname='/home/ats4i/Roboto/Roboto-Regular.ttf', size=16)

countries=["Italy","France","Brazil"]
version=60

for country in countries:
    versionStr=str(version)
    histOptAll= pd.read_table('./results/history_'+country+versionStr+'.csv', sep=",", index_col=0, header=None, 
        names = ["iterations","country","gtot",\
            "s0","startdate","i0","wcases","wrec","r0","d0",\
            "beta","beta2","sigma","sigma2","sigma3","gamma","b","mu"])
    histOptAll=histOptAll.dropna()
    histOptAll = histOptAll.reset_index(drop=True)

    histOpt=histOptAll[histOptAll.gtot==min(histOptAll.gtot)]
    print("---------------------------------------------------------------")
    print("fitting initial conditions")
    print(f"s0={int(histOpt.iloc[0]['s0']+0.5)}, date={int(histOpt.iloc[0]['startdate']+0.5)}, i0={int(histOpt.iloc[0]['i0']+0.5)}, r0={int(histOpt.iloc[0]['r0']+0.5)}, d0={int(histOpt.iloc[0]['d0']+0.5)}, wrec={histOpt.iloc[0]['wrec']:.4f}, wcases={histOpt.iloc[0]['wcases']:.4f}")
    print("parameters")
    print(f"beta={histOpt.iloc[0]['beta']:.8f}, beta2={histOpt.iloc[0]['beta2']:.8f}, 1/sigma={1/float(histOpt.iloc[0]['sigma']):.1f}")
    print(f"1/sigma2={1/float(histOpt.iloc[0]['sigma2']):.1f},1/sigma3={1/float(histOpt.iloc[0]['sigma3']):.1f}, gamma={histOpt.iloc[0]['gamma']:.8f}, b={histOpt.iloc[0]['b']:.8f}")
    print(f"mu={histOpt.iloc[0]['mu']:.8f}, r_0:{(histOpt.iloc[0]['beta']/histOpt.iloc[0]['gamma'])}")
    print("objective function")
    print("f(x)={}".format(histOpt.iloc[0]['gtot'])+" at iteration {}".format(histOpt.index.values[0]))
    print(country+" is done!")

    fig, ax = plt.subplots()
    fte_graph=histOptAll.plot(kind='line',y='gtot',figsize = (12,8))

    x=histOptAll.index[histOptAll.gtot<=(np.mean(histOptAll.gtot)-.4*np.std(histOptAll.gtot))]
    y=histOptAll.gtot[histOptAll.gtot<=(np.mean(histOptAll.gtot)-.4*np.std(histOptAll.gtot))]
    plt.plot(x, y,label="< ($\mu - 0.4 \cdot \sigma$)")

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

    if country=="China":
        date="1/25/20"
    if country=="Italy":
        date="2/24/20"
    if country=="France":
        date="3/3/20"
    if country=="Brazil":
        date="3/02/20"
    
    e0=0
    a0=0
    cleanRecovered=False
    startNCases=100
    predict_range=200
    deltaDate=histOpt.iloc[0]['startdate']
    Date = datetime.strptime(date, "%m/%d/%y")
    end_date = Date + timedelta(days=+int(deltaDate))
    startdate=end_date.strftime('X%m/X%d/%y').replace('X0','X').replace('X','')

    learner = seaird.Learner(country, startdate, predict_range,\
        histOpt.iloc[0].s0, e0, a0, histOpt.iloc[0].i0, histOpt.iloc[0].r0, histOpt.iloc[0].d0, \
            startNCases, histOpt.iloc[0].wcases, histOpt.iloc[0].wrec, cleanRecovered)

    # learner.train()
    # learner.trainPlot()
    death = learner.load_dead(country)
    recovered = learner.load_recovered(country)
    if cleanRecovered:
        zeroRecDeaths=0
    else:
        zeroRecDeaths=1
    data = learner.load_confirmed(country)-zeroRecDeaths*(recovered+death)

    new_index, extended_actual, extended_recovered, extended_death, y0, y1, y2, y3, y4, y5 \
            = learner.predict(histOpt.iloc[0].beta, histOpt.iloc[0].beta2, \
                histOpt.iloc[0].sigma, histOpt.iloc[0].sigma2, histOpt.iloc[0].sigma3, \
                    histOpt.iloc[0].gamma, histOpt.iloc[0].b, histOpt.iloc[0].mu, \
                data, recovered, death, country, histOpt.iloc[0].s0, \
                e0, a0, histOpt.iloc[0].i0, histOpt.iloc[0].r0, histOpt.iloc[0].d0)

    version+=1
    
    df = pd.DataFrame({
                'susceptible': y0,
                'exposed': y1,
                'asymptomatic': y2,
                'infected_data': extended_actual,
                'infected': y3,
                'recovered': extended_recovered,
                'predicted_recovered': y4,
                'death_data': extended_death,
                'predicted_deaths': y5},
                index=new_index)

    color_bg = '#FEF1E5'
    # lighter_highlight = '#FAE6E1'
    darker_highlight = '#FBEADC'
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Novel SEAIR-D Model Results for "+country,
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes, 
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "optimization to find initial conditions",
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                fontproperties=subtitle_font)

    ax.set_ylim((0, max(df.iloc[:]['susceptible'])*1.1))
    df.plot(ax=ax) #,style=['-','-','-','o','-','x','-','s','-'])
    ax.legend(frameon=False)

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
    xy=(1.04, 0.1), xycoords='axes fraction',
    xytext=(0, 0), textcoords='offset points',
    ha='right',rotation=90)
    plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
    xy=(1.045,0.1), xycoords='axes fraction',
    xytext=(0, 0), textcoords='offset points',
    ha='left',rotation=90)

    # Hide grid lines
    ax.grid(False)

    strFile ="./results/modelSEAIRDOptGlobalOptimum"+country+"AdjustICresults2.png"
    fig.tight_layout()
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.show()
    plt.close()

    plotX=df.index[range(0,predict_range)]
    plotXt=df.index[range(0,len(df.infected))]

    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Hide grid lines
    ax.grid(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Zoom at Novel SEAIR-D Model Results for "+country,
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "optimization to find initial conditions",
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                fontproperties=subtitle_font)

    plt.xticks(np.arange(0, predict_range, predict_range/8))
    ax.set_ylim(0,max(df.infected)*1.1)
    ax.plot(plotX,df.infected,'y-',label="Infected")
    ax.plot(plotX,df.predicted_recovered,'c-',label="Recovered")
    ax.plot(plotX,df.predicted_deaths,'m-',label="Deaths")
    ax.plot(plotXt,df.infected_data,'o',label="Infected data")
    ax.plot(plotXt,df.death_data,'x',label="Death data")
    ax.plot(plotXt,df.recovered,'s',label="Recovered data")
    ax.legend(frameon=False)

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
    xy=(1.04, 0.1), xycoords='axes fraction',
    xytext=(0, 0), textcoords='offset points',
    ha='right',rotation=90)
    plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
    xy=(1.045,0.1), xycoords='axes fraction',
    xytext=(0, 0), textcoords='offset points',
    ha='left',rotation=90)

    fig.tight_layout()
    strFile ="./results/ZoomModelSEAIRDOpt"+country+"AdjustICresults2.png"
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.show()
    plt.close()
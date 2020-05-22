import pandas as pd

countries=["Italy","France","Brazil"]

for country in countries:
    histOptAll= pd.read_table('./results/history_'+country+'.csv', sep=",", index_col=0, header=None, 
        names = ["idx","country","gtot",\
            "s0","startdate","i0","wcases","wrec",\
            "beta","beta2","sigma","sigma2","sigma3","gamma","b","mu"])
    histOpt=histOptAll[histOptAll.gtot==min(histOptAll.gtot)]
    print("---------------------------------------------------------------")
    print("fitting initial conditions")
    print(f"s0={int(histOpt.iloc[0]['s0'])}, date={int(histOpt.iloc[0]['startdate'])}, i0={int(histOpt.iloc[0]['i0'])}, wrec={histOpt.iloc[0]['wrec']:.4f}, wcases={histOpt.iloc[0]['wcases']:.4f}")
    print("parameters")
    print(f"beta={histOpt.iloc[0]['beta']:.8f}, beta2={histOpt.iloc[0]['beta2']:.8f}, 1/sigma={1/float(histOpt.iloc[0]['sigma']):.1f}")
    print(f"1/sigma2={1/float(histOpt.iloc[0]['sigma2']):.1f},1/sigma3={1/float(histOpt.iloc[0]['sigma3']):.1f}, gamma={histOpt.iloc[0]['gamma']:.8f}, b={histOpt.iloc[0]['b']:.8f}")
    print(f"mu={histOpt.iloc[0]['mu']:.8f}, r_0:{(histOpt.iloc[0]['beta']/histOpt.iloc[0]['gamma'])}")
    print("objective function")
    print("f(x)={}".format(histOpt.iloc[0]['gtot'])+" at iteration {}".format(histOpt.index.values[0]))
    print(country+" is done!")
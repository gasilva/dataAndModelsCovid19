# dataAndModelsCovid19

## Python code to analyze data and predict Covid-19 infection
 
### Data

This code has data from Repository by Johns Hopkins CSSE

https://github.com/CSSEGISandData/COVID-19

### Data Analysis and Curve Fitting

The log-curve of exponential growth in several countries is based on

https://www.ft.com/content/ae83040c-18e6-440e-bd00-e4c5cefdda26

The growth rate is original from Guilherme A. L. da Silva - http://www.at4i.com

The exponential and logistic curves fitting is based on

https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d

### Mathematical Models

The matematical Models are based in Lotka-Volterra equations, it is like a predator-prey type of model.

#### SIR - Suscetible, Infected and Recovered Model

![](https://user-images.githubusercontent.com/7212952/77828643-775e6a80-70fb-11ea-8428-73d7a2f176a8.gif)

Here, the number of 'recovery' englobes both recovered and deaths. This parameter is represented by γ.

The SIR model code is based on

https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html

https://github.com/Lewuathe/COVID19-SIR

https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

#### SIR-D - SIR model extended to have deaths and recovered separated

The γ is split in two by γ = a + b, where a is the rate of recoveries, and b is the rate of death. Since the death rate seems to be linear (1.5% in China, for example), this linear decomposition of γ is precise enough. 

So we can add a new variable k, (Kill rate), and add to the system of equations. Therefore:

![](https://user-images.githubusercontent.com/7212952/77828649-7f1e0f00-70fb-11ea-8b59-d7f722305847.gif)

The SIR-D model code is based on the contribution of Giuliano Belinassi, from IME-USP, Brazil

https://github.com/Lewuathe/COVID19-SIR/issues/13#issue-589616803

The the Pyhton notebook of

https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model#Scenario-in-Italy





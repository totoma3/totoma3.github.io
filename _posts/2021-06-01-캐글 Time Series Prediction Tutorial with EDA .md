---
layout: single
title: "2차세계대전 데이터와 기온 데이터를 이용한 시계열 분석"
description: "캐글의 노트북 리뷰!"
headline: "캐글의 노트북 리뷰!"
comments: true
published: true
toc: true
toc_label: "Contents"
toc_icon: "cog"
categories:
  - kaggle
tags:
  - machine_learning
  - python
---

# Time Series Prediction Tutorial with EDA


여기 데이터에서는 aerial bombing operations(공중 폭격 작전)과 weather conditions in world war 2(2차 세계대전 날씨)의 데이터를 사용했다.

이 시점 이후에는 2차 세계대전을 약자 ww2로 사용할 것이다.

EDA (Exploratory Data Analysis)과정을 사용할 것이다.

그 후, 우리는 폭격 작업이 언제 완료되는지 예측하는 시계열 예측에 초점을 맞출 것이다.

시계열 예측을 위해 ARIMA 방법을 사용할 것입니다.

**목차**
* 데이터 불러오기
* 데이터 설명
* 데이터 클리닝
* 데이터 시각화
* Time Series Prediction with ARIMA(ARIMA를 사용하여 시계열 예측)
  + What is Time Series ?(시계열은 무엇인가)
  + Stationarity of a Time Series(시계열의 정상성)
  + Make a Time Series Stationary(시계열 정상성 만들기)
    - Moving Average method(이동평균법)
    - Differencing method(차별화 방법)
  + 시계열 예측
* 결론

# 환경 맞추기
```python
pip install plotly==3.10.0
```
```python
#This Python 3 environment comes with many helpful analytics libraries installed
#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
import plotly.plotly as py # visualization library
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object


# import warnings library
import warnings        
# ignore filters
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.
# Any results you write to the current directory are saved as output.
```
# 데이터 불러오기
* Aerial Bombing Operations in WW2
  - 폭격 작전을 포함하는 데이터인데,
    예를들어 1945년 A36 항공기로 독일(베를린) 폰테올리보 비행장 폭탄을 사용한 미국 데이터도 포함한다.
* Wether Conditions in WW2
  - 2차 세계대전동안의 날씨데이터, 예를들어, 조지타운 기상대에 따르면, 평균 기온은 1942년 1/7에서 23.88라는 자료이다.
  - 이 데이터는 2개의 하위 집합이 있다. 첫 번째는 국가, 위도, 경도와 같은 기상 관측소 위치, 두번째는 기상 관측소에서 측정한 최소, 최대 및 평균 온도이다.

**kaggle에서는 바로 불러올 수 있으나 연습으로 할 때 따로 데이터 셋을 저장하여 주피터 노트북에서 사용하였다.**

```python
# bombing data
aerial = pd.read_csv("C:\kaggle\input\operations.csv")
# first weather data that includes locations like country, latitude and longitude.
weather_station_location = pd.read_csv("C:\kaggle\input\Weather Station Locations.csv")
# Second weather data that includes measured min, max and mean temperatures
weather = pd.read_csv("C:\kaggle\input\Summary of Weather.csv")
```

# 데이터셋 살펴보기
```python
aerial.head()
```
![kaggle_output1](https://user-images.githubusercontent.com/79041564/120348716-64b37700-c338-11eb-9348-11df9f0e8936.png)


```python
weather.head()
```

![kaggle_output2](https://user-images.githubusercontent.com/79041564/120348725-6846fe00-c338-11eb-81ca-0eb7d12288dd.png)

# 데이터 설명
* Aerial bombing Data 설명
  - Mission Date: 미션의 날짜
  - Theater of Operations: 현재 군사작전이 진행 중인 지역
    "군대는 현장에서 작전을 기다리고 있었다."를 예를 들면: "베트남 극장에서 3년간 근무했습니다"
  - Country: 미국처럼 임무나 작전을 수행하는 나라
  - Air Force: 5AF와 같은 공군 통합의 명칭 또는 ID
  - Aircraft Series: B24와 같은 항공기 모델 또는 유형
  - Callsign: 폭탄 공격전에 메세지나 코드, 방송, 라디오로 알린다.
  - Takeoff Base: "폰테 올리보 비행장"처럼 이륙 공항 이름
  - Takeoff Location: Sicily의 이륙 지역
  - Takeoff Latitude: 이륙 지역의 위도
  - Takeoff Longitude: 이륙 지역의 경도
  - Target Country: "독일" 같은 목표 국가
  - Target City: "베를린" 같은 목표 도시
  - Target Type: "도시지역" 같은 목표의 유형
  - Target Industry: 도시나 도회지같은 목표 산업
  - Target Priority: 1(가장 높음)과 같은 목표 우선순위
  - Target Latitude: 목표의 위도
  - Target Longitude: 목표의 경도

* Weather Condition data 설명
  - Weather station location(기상대 위치)
    + WBAN: 기상청 번호
    + NAME: 기상 관측소 이름
    + STATE/COUNTRY ID: 국가의 약자
    + Latitude: 기상 관측소의 위도
    + Longitude: 기상대의 경도
  - Weather
    + STA: 어느 역 번호 (WBAN) (네이버 어휘사전)
    + Date: 온도측정일자
    + MeanTemp: 평균 온도


# 데이터 클리닝
* Aerial Bombing 데이터는 NaN을 많이 포함하고 있다. 여기서 NaN을 drop했다.
  이렇게 함으로써 불확실성을 제거할 뿐만 아니라 시각화 과정도 간소화 되었다.
  - Drop countries that are NaN(NaN인 국가 삭제)
  - Drop if target longitude is NaN(NaN인 타겟 경도 삭제)
  - Drop if takeoff longitude is NaN(NaN인 이륙 경도 삭제)
  - Drop unused features(사용하지 않는 기능 삭제)
* 여기서 Weather Condition 데이터는 클리닝할 필요가 없어서 놔두었다.
  그러나 데이터 변수는 우리가 사용하는 것만 넣을 것이다.

```python
# NaN인 국가 삭제
aerial = aerial[pd.isna(aerial.Country)==False]
# NaN인 타겟 경도 삭제
aerial = aerial[pd.isna(aerial['Target Longitude'])==False]
# NaN인 이륙 경도 삭제
aerial = aerial[pd.isna(aerial['Takeoff Longitude'])==False]
# 사용하지 않는 기능 삭제
drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',
             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',
             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',
             'High Explosives', 'High Explosives Type','Mission Type',
             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',
             'Incendiary Devices', 'Incendiary Devices Type',
             'Incendiary Devices Weight (Pounds)',
             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',
             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',
             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',
             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']
aerial.drop(drop_list, axis=1,inplace = True)
aerial = aerial[ aerial.iloc[:,8]!="4248"] # 이 이륙 위도 삭제 
aerial = aerial[ aerial.iloc[:,9]!=1355]   # 이 이륙 경도 삭제
```  
  
# 데이터 클리닝 후 데이터셋 살펴보기
```python
aerial.info()
```

![kaggle_output3](https://user-images.githubusercontent.com/79041564/120349061-bb20b580-c338-11eb-94a6-3dc06448f96f.png)

# 우리가 weather_station_location에서 사용할 것만 고르기
```python
weather_station_location = weather_station_location.loc[:,["WBAN","NAME","STATE/COUNTRY ID","Latitude","Longitude"] ]
weather_station_location.info()
```

![kaggle_output4](https://user-images.githubusercontent.com/79041564/120349277-eefbdb00-c338-11eb-84c8-fb5d86f83a89.png)

```python
# 마찬가지로 우리가 weather에서 우리가 사용할 것만 고르기
weather = weather.loc[:,["STA","Date","MeanTemp"] ]
weather.info()
```

![kaggle_output5](https://user-images.githubusercontent.com/79041564/120349336-fe7b2400-c338-11eb-899d-bec3baf3b7e6.png)


# 데이터 시각화
* 데이터를 이해하기 위해 시각화를 해보자
  - 얼마나 많은 나라를 공격했나
  - 상위 타겟 국가
  - 상위 10개의 aircraft series(항공기 모델,유형)
  - 이륙 기지 위치(공격하는 나라들)
  - 타겟 위치
  - 폭파 경로
  - 현재 군사작전이 진행 중인 지역
  - 기상 관측소 위치


```python
# 임무나 작전을 수행하는 나라
print(aerial['Country'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(aerial['Country'])
plt.show()
```

![kaggle_output6](https://user-images.githubusercontent.com/79041564/120349912-79dcd580-c339-11eb-8518-5763f926a84e.png)

```python
# 상위 타겟 국가들
print(aerial['Target Country'].value_counts()[:10])
plt.figure(figsize=(22,10))
sns.countplot(aerial['Target Country'])
plt.xticks(rotation=90)
plt.show()
```

![kaggle_output7](https://user-images.githubusercontent.com/79041564/120349980-882af180-c339-11eb-9610-8e1c484f9143.png)

```python
# 항공기 모델,유형
data = aerial['Aircraft Series'].value_counts()
print(data[:10])
data = [go.Bar(
            x=data[:10].index,
            y=data[:10].values,
            hoverinfo = 'text',
            marker = dict(color = 'rgba(177, 14, 22, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    )]

layout = dict(
    title = 'Aircraft Series',
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```

![kaggle_output8](https://user-images.githubusercontent.com/79041564/120350039-9547e080-c339-11eb-993c-b88f918e6ec6.png)

```python
# 공격
aerial["color"] = ""
aerial.color[aerial.Country == "USA"] = "rgb(0,116,217)"
aerial.color[aerial.Country == "GREAT BRITAIN"] = "rgb(255,65,54)"
aerial.color[aerial.Country == "NEW ZEALAND"] = "rgb(133,20,75)"
aerial.color[aerial.Country == "SOUTH AFRICA"] = "rgb(255,133,27)"

data = [dict(
    type='scattergeo',
    lon = aerial['Takeoff Longitude'],
    lat = aerial['Takeoff Latitude'],
    hoverinfo = 'text',
    text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = aerial["color"],
        opacity = 0.7),
)]
layout = dict(
    title = 'Countries Take Off Bases ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='Mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```

![kaggle_output9](https://user-images.githubusercontent.com/79041564/120350114-a55fc000-c339-11eb-8165-475f63d31a2d.png)


**이제 공격하는 나라에서 이륙해서 폭탄을 어느 나라로 떨어뜨리는지 폭탄 경로를 시각화해보자.**

```python
# 폭탄 경로
# 경로1
airports = [ dict(
        type = 'scattergeo',
        lon = aerial['Takeoff Longitude'],
        lat = aerial['Takeoff Latitude'],
        hoverinfo = 'text',
        text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
        mode = 'markers',
        marker = dict( 
            size=5, 
            color = aerial["color"],
            line = dict(
                width=1,
                color = "white"
            )
        ))]
# 경로2
targets = [ dict(
        type = 'scattergeo',
        lon = aerial['Target Longitude'],
        lat = aerial['Target Latitude'],
        hoverinfo = 'text',
        text = "Target Country: "+aerial["Target Country"]+" Target City: "+aerial["Target City"],
        mode = 'markers',
        marker = dict( 
            size=1, 
            color = "red",
            line = dict(
                width=0.5,
                color = "red"
            )
        ))]
        
# 경로3
flight_paths = []
for i in range( len( aerial['Target Longitude'] ) ):
    flight_paths.append(
        dict(
            type = 'scattergeo',
            lon = [ aerial.iloc[i,9], aerial.iloc[i,16] ],
            lat = [ aerial.iloc[i,8], aerial.iloc[i,15] ],
            mode = 'lines',
            line = dict(
                width = 0.7,
                color = 'black',
            ),
            opacity = 0.6,
        )
    )
    
layout = dict(
    title = 'Bombing Paths from Attacker Country to Target ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='Mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
    
fig = dict( data=flight_paths + airports+targets, layout=layout )
iplot( fig )
```

![kaggle_output10](https://user-images.githubusercontent.com/79041564/120350240-c3c5bb80-c339-11eb-87e9-9bf2c7295c6b.png)

그림에서 볼 수 있듯이 대부분의 폭격이 지중해의 Theater of Operations에서 벌어진다.
* ETO: European Theater of Operations(유럽에서의 작전 지역)
* PTO: Pasific Theater of Operations(태평양의 작전 지역)
* MTO: Mediterranean Theater of Operations(지중해의 작전 지역)
* CBI: China-Burma-India Theater of Operations(버마 작전 지역)
* EAST AFRICA: East Africa Theater of Operations(동아프리카 작전 지역)

**작전 지역을 그래프로 나타내어 보자!**

```python
#Theater of Operations(작전 지역)
print(aerial['Theater of Operations'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(aerial['Theater of Operations'])
plt.show()
```

![kaggle_output11](https://user-images.githubusercontent.com/79041564/120351036-77c74680-c33a-11eb-80d4-c5878a48511c.png)

# Weather station location(기상소 위치)

```python
# weather station locations(기상소 위치)

data = [dict(
    type='scattergeo',
    lon = weather_station_location.Longitude,
    lat = weather_station_location.Latitude,
    hoverinfo = 'text',
    text = "Name: " + weather_station_location.NAME + " Country: " + weather_station_location["STATE/COUNTRY ID"],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 8 ,
        line = dict(width=1,color = "white"),
        color = "blue",
        opacity = 0.7),
)]
layout = dict(
    title = 'Weather Station Locations ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='Mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```

![kaggle_output12](https://user-images.githubusercontent.com/79041564/120351118-8d3c7080-c33a-11eb-8451-27f4a99bd8b8.png)


* USA와 BURMA 전쟁에 초점을 맞춰보자
* 이 전쟁에서 미국은 1942년부터 1945년까지 버마를 폭격했다.
* 이 전쟁에서 가장 가까운 기상대는 BINDUKURI이고 1943년부터 1945년까지의 기온 기록을 가지고 있다.
* 이제 이 상황을 시각화해 보자. 시각화하기 전에 날짜 특징 객체를 만들자.


```python
weather_station_id = weather_station_location[weather_station_location.NAME == "BINDUKURI"].WBAN 
#WBAN은 기상청 번호
weather_bin = weather[weather.STA == 32907]
#STA는 기상청 번호(WBAN)
weather_bin["Date"] = pd.to_datetime(weather_bin["Date"])
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.show()
```

![kaggle_output13](https://user-images.githubusercontent.com/79041564/120351187-a0e7d700-c33a-11eb-9f84-2e1bb65d7bdf.png)

* 데이터셋 정보를 보고 알 수 있듯이 온도 측정은 1943년부터 1945년까지있다.
* 온도는 12도에서 32도 사이에서 왔다갔다한다.
* 온도를 보고 알 수 있듯이 겨울은 여름보다 춥다.

```python
aerial = pd.read_csv("C:\kaggle\input\operations.csv")
aerial["year"] = [ each.split("/")[2] for each in aerial["Mission Date"]]
aerial["month"] = [ each.split("/")[0] for each in aerial["Mission Date"]]
aerial = aerial[aerial["year"]>="1943"]
aerial = aerial[aerial["month"]>="8"]

aerial["Mission Date"] = pd.to_datetime(aerial["Mission Date"])

attack = "USA"
target = "BURMA"
city = "KATHA"

aerial_war = aerial[aerial.Country == attack]
aerial_war = aerial_war[aerial_war["Target Country"] == target]
aerial_war = aerial_war[aerial_war["Target City"] == city]
```

```python
liste = []
aa = []
for each in aerial_war["Mission Date"]:
    dummy = weather_bin[weather_bin.Date == each]
    liste.append(dummy["MeanTemp"].values)
aerial_war["dene"] = liste
for each in aerial_war.dene.values:
    aa.append(each[0])

# Create a trace
trace = go.Scatter(
    x = weather_bin.Date,
    mode = "lines",
    y = weather_bin.MeanTemp,
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
    name = "Mean Temperature"
)
trace1 = go.Scatter(
    x = aerial_war["Mission Date"],
    mode = "markers",
    y = aa,
    marker = dict(color = 'rgba(16, 0, 200, 1)'),
    name = "Bombing temperature"
)
layout = dict(title = 'Mean Temperature --- Bombing Dates and Mean Temperature at this Date')
data = [trace,trace1]

fig = dict(data = data, layout = layout)
iplot(fig)
```

![kaggle_output14](https://user-images.githubusercontent.com/79041564/120351318-c1179600-c33a-11eb-890a-e37907e19ff5.png)

* 녹색 선은 Bindukuri에서 측정한 평균 온도입니다.
* 파란색 표시는 폭격 날짜 및 폭격이 벌어지는 날의 온도입니다.
* USA는 높은 온도일때 폭격을 했다.
  - 문제는 우리가 미래의 날씨를 예측할 수 있고, 이 예측에 따라 우리는 폭격이 행해질 것인지 아닌지를 알 수 있다는 것이다.
  - 이 질문에 답하려면 먼저 시계열 예측부터 시작해야 한다.

# ARIMA와 함께 시계열 예측
* 우리는 ARIMA 모델을 사용할 것이다.
* ARIMA : AutoRegressive Integrated Moving Average(자동 회귀 통합 이동 평균)
* 이 방법에 대해 알아보려면 다음을 알아야한다.:
  - 시계열이 무엇인가?
  - 시계열의 정상성은 무엇인가?
  - 시계열의 정상성을 만들어야하나?
  - 시계열 예측


# 시계열이란 무엇인가?
* 시계열은 일정한 시간 간격으로 수집된 데이터 점의 집합이다.
* 시계열은 시간에 의존한다.
* 대부분의 시계열은 계절적 경향의 형태를 가지고 있다. 예를 들어, 우리가 아이스크림을 판매한다면, 아마도 여름 시즌에는 더 높은 판매가 있을 것이다. 따라서, 이 시계열은 계절성 추세를 가지고 있다고 할 수 있다.
* 또 다른 예로, 1년 동안 매일 한 번씩 주사위를 던졌다고 생각해보면 숫자 6이 주로 여름 시즌에 등장하거나 숫자5가 대부분 1월에 등장한다는 식의 시나리오는 없을 것이다. 따라서 이 시계열에는 계절성 추세가 없다고 할 수 있다.

# 시계열의 정상성
* 정상성인지 아닌지 이해하기 위해서 3가지의 기본 조건이 있다.
  - 평균같은 시계열의 통계학적인 특성에서 시간이 경과함에도 분산은 계속 유지되어야한다.
    + 상수의 평균
    + 상수의 분산
    + 자기공분산(autocovariance)은 시간에 의존하지 않는다. 자기 공분산은 시계열과 lagged(지연) 시계열 간의 공분산입니다.

**데이터에서 시각화와 계절성을 체크해보자!**

```python
# Bindikuri지역의 평균 온도
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.show()

# weather에서부터 시계열을 만들어 보자
timeSeries = weather_bin.loc[:, ["Date","MeanTemp"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
```

![kaggle_output15](https://user-images.githubusercontent.com/79041564/120351549-f0c69e00-c33a-11eb-98a6-026abed12c93.png)

* 그래프를 보고 알 수 있듯이 데이터의 시계열은 계절적 변동을 가지고 있다. 여름에는 평균 기온이 더 높고 겨울에는 평균 기온이 매년 더 낮다.
* 이제 정상성을 체크해보자. 우리는 다음 방법으로 정상성을 체크할 수 있다.
  - Plotting Rolling Statistics: window 크기를 6이라고 하는 window를 가지고 있고 그리고 우리는 정상성을 체크할 수 있는 rolling mean와 variance를 발견할 수 있다.
  - Dickey-Fuller Test: 검정 결과는 검정 통계량과 차이 신뢰 수준에 대한 일부 임계 값으로 구성된다. 검정 통계량이 임계값보다 작으면 시계열이 정지 상태라고 말할 수 있다.

```python
# adfuller library, ADF검정을 사용할 수 있는 파이썬 패키지
from statsmodels.tsa.stattools import adfuller
# check_adfuller
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
# check_mean_std

def check_mean_std(ts):
    #Rolling statistics
    #여기서 pd.rolling_mean(ts, window=6)를  ts.rolling(6).mean()으로 바꿔줘야한다.
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
    
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts)
check_adfuller(ts.MeanTemp)
```

![kaggle_output16](https://user-images.githubusercontent.com/79041564/120351649-04720480-c33b-11eb-8d4b-106ee3db6bc6.png)


* 정상성의 첫 번째 기준은 상수 평균이다.따라서 위의 그림(검은색 선)에서 볼 수 있듯이 평균이 일정하지 않기 때문에 정상성이 아님을 알 수 있다. (정상성 아님)
* 두 번째 초록색 그래프는 상수 분산이다. 그래프를 보면 일정해보인다.(정상성임)
* 세 번째 방법은 검정 통계량이 임계값보다 작을 경우 시계열은 정상성이라고 말할 수 있다. 결과값을 보면 t-test값이 -1.4이고 Critical 값이 1%일때 -3.4, 5%일때 -2.8, 10%일때 -2.5임을 알 수 있다. 즉 t-테스트 값이 Critical 값보다 크다.
* 결과적으로, 우리는 데이터의 시계열이 정상성이 아니라고 확신한다.
* 다음 파트에서 정상성으로 만들어 보자!

# 시계열의 정상성 만들기?
* 위에서 밝혀진 것처럼 이 데이터가 정상성이 아닌 이유가 2가지 있다.
  - Trend(추세): 시간 경과에 따른 평균. 즉 우리는 시계열의 정상성을 위해 일정한 평균이 필요하다.
  - Seasonality(계절성): 특정 시점의 변동. 즉 우리는 시계열의 정상성을 위해 일정한 변형이 필요하다.
* 첫번째로 추세문제를 풀어보자!
  - 가장 유명한 방법은 moving average방법이다.
    + Moving average: 우리는 'n'개 샘플의 평균을 구하는 창이 있다. 'n'은 창 크기이다.

```python
# Moving average method(이동 평균 방법)
window_size = 6
moving_avg = ts.rolling(window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(ts, color = "red",label = "Original")
plt.plot(moving_avg, color='black', label = "moving_avg_mean")
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.show()    
```

![kaggle_output17](https://user-images.githubusercontent.com/79041564/120351827-28cde100-c33b-11eb-9534-f183f154049e.png)

```python
ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True) # 첫 번째 6은 창 크기로 인해 nan 값입니다.

# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_moving_avg_diff)
check_adfuller(ts_moving_avg_diff.MeanTemp)
```

![kaggle_output18](https://user-images.githubusercontent.com/79041564/120351884-34210c80-c33b-11eb-9954-a479e077779e.png)

Test statistic:  -11.13851433513848
p-value:  3.150868563164539e-20
Critical Values: {'1%': -3.4392539652094154, '5%': -2.86546960465041, '10%': -2.5688625527782327}

* 상수 평균 기준: 위의 그림(검은색 선)에서 볼 수 있듯이 평균은 상수처럼 보인다. (정상성 조건 충족)
* 두 번째는 상수 분산인데 일정해 보인다. (정상성 충족)
* 검정 통계량은 1%의 임계 값보다 작으므로 99%의 신뢰도로 정상성이라고 말할 수 있습니다. (정상성 조건 충족)
  - Differencing method: 시계열과 이동 시계열 간의 차이를 갖는 것이 좋습니다.







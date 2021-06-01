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

```python
weather.head()
```






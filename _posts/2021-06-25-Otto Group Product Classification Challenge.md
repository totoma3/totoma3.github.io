---
layout: single
title: "Otto Group Product Classification Challenge"
description: "Otto Group Product Classification Challenge 성능 비교"
headline: "Otto Group Product Classification Challenge"
comments: true
published: true
toc: true
toc_label: "Contents"
toc_icon: "cog"
categories:
  - Deep_learning
tags:
  - machine_learning
  - Deep_learning
  - python
---


# 파일 불러오기
```python
#파일 불러오기
import pandas as pd
df = pd.read_csv("C:/kaggle/otto-group-product-classification-challenge/train.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>...</th>
      <th>feat_85</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>61873</th>
      <td>61874</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>Class_9</td>
    </tr>
    <tr>
      <th>61874</th>
      <td>61875</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Class_9</td>
    </tr>
    <tr>
      <th>61875</th>
      <td>61876</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_9</td>
    </tr>
    <tr>
      <th>61876</th>
      <td>61877</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>Class_9</td>
    </tr>
    <tr>
      <th>61877</th>
      <td>61878</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>Class_9</td>
    </tr>
  </tbody>
</table>
<p>61878 rows × 95 columns</p>
</div>




```python
#id는 필요없으므로 제거해준다.
df=df.drop(['id'],axis=1)
```


```python
#수치로 바꿔주기위해 LabelEncoder을 사용한다.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['target']=le.fit_transform(df['target'])
```

# 시각화로 확인
```python
#시각화로 확인
#타겟이 class1~9로 써있던것이 숫자 0부터 8로 바뀌었다.
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
sns.countplot(df['target'])
```

    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(<AxesSubplot:xlabel='target', ylabel='count'>

   

# 트레이닝과 테스트를 위한 모델 준비

```python
#트레이닝과 테스트를 위한 모델 준비
y=df['target']
x=df.drop(['target'],axis=1)
```


```python
#train 데이터(0.8)와 test데이터(0.2)로 나누기
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
```

# 로지스틱 회귀와 정확도 점수 측정을 위해 패키지 불러오기
```python
#로지스틱 회귀와 정확도 점수 측정을 위해 패키지 불러오기
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
list_models=[]
list_scores=[]
lr=LogisticRegression(max_iter=100000) # max_iter=Gradient Descent 방식을 반복해서 몇번 수행할 것인가
lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)
list_models.append('logistic regression')
list_scores.append(score_1)
```


```python
fig,axes=plt.subplots(1,2)
fig.set_size_inches(11.7, 8.27)
sns.countplot(pred_1,ax=axes[0])
sns.countplot(y_test,ax=axes[1])
```

    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='target', ylabel='count'>




    
![output_8_2](https://user-images.githubusercontent.com/79041564/123429906-7835c300-d602-11eb-816c-bf516b62227e.png)

    



```python
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)
score_2=accuracy_score(y_test,pred_2)
list_scores.append(score_2)
list_models.append('random forest classifier')
```


```python
score_2
```
    0.8062378797672916

```python
fig,axes=plt.subplots(1,2)
fig.set_size_inches(11.7, 8.27)
sns.countplot(pred_2,ax=axes[0])
sns.countplot(y_test,ax=axes[1])
```

    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn( <AxesSubplot:xlabel='target', ylabel='count'>




    
![output_11_2](https://user-images.githubusercontent.com/79041564/123429922-7ec43a80-d602-11eb-818c-de1ed7c6cebc.png)

    


# 로지스틱 회귀 모형과 랜덤 포리스트 모형에 대한 비교 bw 예측을 생성해 봅시다.
```python
#로지스틱 회귀 모형과 랜덤 포리스트 모형에 대한 비교 bw 예측을 생성해 봅시다.
fig,axes=plt.subplots(1,2)
fig.set_size_inches(11.7, 8.27)
sns.countplot(pred_1,ax=axes[0])
axes[0].legend(title='predictions by logistic regression')
sns.countplot(pred_2,ax=axes[1])
axes[1].legend(title='predictions by random forest')
```

    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    No handles with labels found to put in legend.
    C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    No handles with labels found to put in legend.
    




    <matplotlib.legend.Legend at 0x126d01a6bb0>




    
![output_12_2](https://user-images.githubusercontent.com/79041564/123429936-82f05800-d602-11eb-9de6-f930a3063259.png)

    


위의 관측에서, 우리는 이러한 예측들의 유일한 주요 차이 bw는 로지스틱 회귀 분석의 예측과 비교하여 랜덤 포리스트에서 1 클래스의 카운트가 더 적고 2 클래스의 카운트가 더 높다는 결론을 내릴 수 있다.


```python
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
pred_3=svm.predict(x_test)
score_3=accuracy_score(y_test,pred_3)
list_scores.append(score_3)
list_models.append('support vector machines')
```


```python
score_3
```




    0.7798965740142211




```python
!pip install xgboost 
```

    Collecting xgboost
      Downloading xgboost-1.4.2-py3-none-win_amd64.whl (97.8 MB)
    Requirement already satisfied: scipy in c:\users\administrator\anaconda3\lib\site-packages (from xgboost) (1.5.2)
    Requirement already satisfied: numpy in c:\users\administrator\anaconda3\lib\site-packages (from xgboost) (1.19.2)
    Installing collected packages: xgboost
    Successfully installed xgboost-1.4.2
    


```python
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
pred_4=xgb.predict(x_test)
score_4=accuracy_score(y_test,pred_4)
list_models.append('xgboost classifier')
list_scores.append(score_4)
```

    C:\Users\Administrator\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [20:34:09] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    


```python
score_4
```




    0.8122979961215255




```python
plt.figure(figsize=(12,5))
plt.bar(list_models,list_scores,width=0.3)
plt.xlabel('classifictions models')
plt.ylabel('accuracy scores')
plt.show()
```


    
![output_19_0](https://user-images.githubusercontent.com/79041564/123429953-88e63900-d602-11eb-8e93-a7ce915595d5.png)

    


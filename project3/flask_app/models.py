# 모듈 설치 안되어 있으면 pip 이용해서 설치하기
# pip install --upgrade pandas 
# pip install scikit-learn
import pandas as pd
import numpy as np
import pickle
import scipy.io
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 파일 불러오기 (해당 폴더안에 데이터 넣어줌)
df = pd.read_csv('netflex_data.csv')
# 실행이 잘되는지 print 문 이용해서 확인하기 
# print(df)

# 범주형 변수를 삭제하거나 원핫 인코딩 등으로 전처리하면 좋다
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['type', 'title','cast','country',	'date_added',	'rating',	'duration',	'listed_in',	'description']

for col in cols:
  df[col] = le.fit_transform(df[col])


from sklearn.model_selection import train_test_split

# train / validation / test 데이터셋으로 분리
train, test = train_test_split(df, test_size=0.2, random_state=1)
train, val = train_test_split(train, test_size=0.25, random_state=1)

target = 'rating'
features = df.columns.drop(['rating','show_id'])

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(df[target].value_counts(normalize=True))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#기준모델(y = average)
predict = y_train.mean()
y_pred = [predict] * len(y_train)

#기준모델 성능
print('*기준모델 성능')
print('mae :', mean_absolute_error(y_train, y_pred))
print('mse :', mean_squared_error(y_train, y_pred))
print('r2  :', r2_score(y_train, y_pred))

# 모듈 빠진거 있음 설치 : pip install category_encoders
# 몯류
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold  # 데이터 분할, KFold
from sklearn.linear_model import LogisticRegression          # 1. 로지스틱 회귀 => 정규화 필수(StandardScaler)
from sklearn.tree import DecisionTreeClassifier              # 2. 의사결정나무 - 분류
from sklearn.ensemble import RandomForestClassifier          # 3. 랜덤포레스트 - 분류                                    # 4. Xgboost
from sklearn.svm import SVC                                  # 5. 서포트 벡터 머신
from sklearn.neighbors import KNeighborsClassifier           # 6. K-최근접 이웃 분류
from sklearn.metrics import roc_curve, roc_auc_score, auc    # roc_auc_score
from datetime import datetime                
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# 파이프라인 생성 및 학습
pipe = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(), 
    RandomForestRegressor(random_state=1, n_jobs=-1)
)

pipe.fit(X_train, y_train)

k = 5
scores_mae = cross_val_score(pipe, X_train, y_train, cv=k, scoring='neg_mean_absolute_error')
scores_mse = cross_val_score(pipe, X_train, y_train, cv=k, scoring='neg_mean_squared_error')
scores_r2 = cross_val_score(pipe, X_train, y_train, cv=k, scoring='r2')

print('*적용 모델 성능')
print(f'mae:', scores_mae)
print('평균 mae:', scores_mae.mean())
print(f'mse:', scores_mse)
print('평균 mse:', scores_mse.mean())
print(f'r2:', scores_r2)
print('평균 r2:', scores_r2.mean())

y_pred = pipe.predict(X_test)
print(f'mae:', mean_absolute_error(y_test, y_pred))
print(f'mse:', mean_squared_error(y_test, y_pred))
print(f'r2:', r2_score(y_test, y_pred))



#pickle 저장
with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(pipe, pickle_file)


joblib.dump(pipe, 'model/model.pkl')




y_pred = pipe.predict(X_test)


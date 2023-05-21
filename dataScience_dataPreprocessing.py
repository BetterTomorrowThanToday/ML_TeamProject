from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("서울교통공사_역별 일별 시간대별 승하차인원 정보_20221231.csv", encoding="cp949")
data['고유역번호(외부역코드)'] = data['고유역번호(외부역코드)'].apply(lambda x: int(x) if str(x).isdigit() else x)

grouped_data = data.groupby('고유역번호(외부역코드)')
station_id = 215

# 잠실나루역
filtered_data = grouped_data.get_group(station_id)

# 홀수가 승차, 짝수가 하차
onBoard_data = filtered_data.iloc[::2]
offBoard_data = filtered_data.iloc[1::2]


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

#날짜 int 변환
y_raw = onBoard_data['수송일자'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d').strftime('%m%d')))
passenger = onBoard_data[['06시이전']].iloc[::7]
date = y_raw.iloc[::7]
#
# .iloc[::7]
date= np.array(date)
passenger = np.array(passenger)

train_input, test_input, train_target, test_target = train_test_split(date, passenger, test_size=0.2, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
reg = LinearRegression()
reg.fit(train_input, train_target)

# 모델의 정확도
print("train score: ", reg.score(train_input,train_target))
print("test score: ", reg.score(test_input, test_target))

test_date = date.reshape(-1,1)
pred = reg.predict(test_date)

plt.scatter(train_input, train_target, marker="^")
plt.scatter(date, pred, color='#00ff00')
plt.xlabel("date")
plt.ylabel("passenger")
plt.title("LINEAR REGRESSION")
plt.show()

# print("예상 탑승객 :", pred)
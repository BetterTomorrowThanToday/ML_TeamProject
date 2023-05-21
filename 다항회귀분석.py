import datetime
from datetime import datetime as dateTime
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from sklearn.preprocessing import PolynomialFeatures


# csv import
data = pd.read_csv("서울교통공사_역별 일별 시간대별 승하차인원 정보_20221231.csv", encoding="cp949")

# change code to int
data['고유역번호(외부역코드)'] = data['고유역번호(외부역코드)'].apply(lambda x: int(x) if str(x).isdigit() else x)

# 역 구분
grouped_data = data.groupby('고유역번호(외부역코드)')

# 잠실나루역 2022 데이터
station_id = 215
잠실나루_data = grouped_data.get_group(station_id)

# 수송일자를 기준으로 요일 열 추가
잠실나루_data['수송일자'] = pd.to_datetime(잠실나루_data['수송일자'])
잠실나루_data['요일'] = 잠실나루_data['수송일자'].dt.dayofweek

# 숫자를 요일로 변경
# 잠실나루_data.loc[잠실나루_data['요일'] == 0, '요일'] = '월요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 1, '요일'] = '화요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 2, '요일'] = '수요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 3, '요일'] = '목요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 4, '요일'] = '금요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 5, '요일'] = '토요일'
# 잠실나루_data.loc[잠실나루_data['요일'] == 6, '요일'] = '일요일'

# 월 추가
잠실나루_data['월'] = 잠실나루_data['수송일자'].dt.month

# 수송일자 데이터를 int 형식으로 변환
잠실나루_data['수송일자'] = pd.to_numeric(잠실나루_data['수송일자'])
# 잠실나루_data['수송일자'] = 잠실나루_data['수송일자'].apply(lambda x: int(dateTime.strptime(x, '%Y-%m-%d').strftime('%m%d')))


# 홀수가 승차, 짝수가 하차
onBoard_data = 잠실나루_data.iloc[::2]
offBoard_data = 잠실나루_data.iloc[1::2]

# # 날짜 int 변환
# y_raw = onBoard_data['수송일자'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d').strftime('%m%d')))
# date = y_raw

# 날짜 데이터, 06시 이전 탑승객 데이터
passenger = onBoard_data[['06시이전']]
infos = onBoard_data[['요일','월']]

infos= np.array(infos)
passenger = np.array(passenger)



train_input, test_input, train_target, test_target = train_test_split(infos, passenger, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3, include_bias=False)

train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

reg = LinearRegression()
model = reg.fit(train_poly, train_target)

# 모델의 정확도
print("train score: ", reg.score(train_poly,train_target))
print("test score: ", reg.score(test_poly, test_target))

# 그래프 그리기

# 독립 변수 범위 설정
x_pred = np.linspace(min(infos[:, 0]), max(infos[:, 0]), 10)
y_pred = np.linspace(min(infos[:, 1]), max(infos[:, 1]), 10)
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

# 예측 수행
predicted = model.predict(poly.transform(model_viz))

x = infos[:, 0]  # Accessing the first column (요일)
y = infos[:, 1]  # Accessing the second column (월)
z = passenger.flatten()  # Flatten the passenger array

plt.style.use('default')

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)  # 검은색 마커들
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, s=20, edgecolor='#0000ff')  # 파란색 마커들
    ax.set_xlabel('weekday', fontsize=12)  # 해당 축을 설명하는 라벨
    ax.set_ylabel('month', fontsize=12)
    ax.set_zlabel('passenger', fontsize=12)
    ax.locator_params(nbins=6, axis='x')  # 해당 축의 구간 개수 (쪼개진 구간도 포함)
    ax.locator_params(nbins=12, axis='y')
    ax.locator_params(nbins=4, axis='z')

# 높이와 방위각을 조절해서 보고 싶은 위치를 조정할 수 있음
# elevation (높이), azimuth (방위각)
ax1.view_init(elev=27, azim=112)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)
fig.tight_layout()

# 그래프 출력
plt.title("POLYNOMIAL REGRESSION")
plt.show()
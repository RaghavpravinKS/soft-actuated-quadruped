import numpy as np  
import pandas as pd
import torch 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#data = pd.read_excel("height_angle_data.xlsx")

# x = data['angle']
# y = data['height']

data= pd.read_excel("height_angle_data.xlsx")

average_height_df = data.groupby(['angle', 'direction'])['height'].mean().reset_index()
average_height_df.columns = ['angle', 'direction', 'average_height']

height_down = average_height_df[average_height_df['direction'] == 0]
height_up = average_height_df[average_height_df['direction'] != 0]

#average_height_df = pd.concat([average_height_df, height_down],ignore_index= True)

#print(average_height_df)
window_size = 3 # Adjust the window size as needed
height_down['smoothed_height'] = height_down['average_height'].rolling(window=window_size, min_periods=1).mean()

window_size = 6 # Adjust the window size as needed
height_down['double_smoothed_height'] = height_down['average_height'].rolling(window=window_size, min_periods=1).mean()

window_size = 3 # Adjust the window size as needed
height_up['smoothed_height'] = height_up['average_height'].rolling(window=window_size, min_periods=1).mean()

window_size = 6 # Adjust the window size as needed
height_up['double_smoothed_height'] = height_up['average_height'].rolling(window=window_size, min_periods=1).mean()

plt.plot(height_up['angle'], height_up['double_smoothed_height'], color = 'red')
plt.plot(height_down['angle'], height_down['double_smoothed_height'])
plt.show()

print(height_down)
pred = pd.concat([height_up.iloc[::-1], height_down], ignore_index=True)
pred.to_csv('average_height_data_up_down.csv')
#average_height_df.to_csv('average_height_data_up_down.csv')



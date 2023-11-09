import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data= pd.read_excel("height_angle_data.xlsx")

x = data['angle'].to_numpy()
d = data['direction'].to_numpy()
y = data['height'].to_numpy()

avg_df = pd.read_csv('average_height_data.csv')

index = np.where(d == 1)
x = x[index]
y = y[index]


angles = np.reshape(np.linspace(40, 180, 141), (141,1))
ups    = np.ones((141,1))

combined_model_pred = pd.read_csv('Leg_combined_model\predictions.csv')
experimental_model_pred = pd.read_csv('Leg_experimental_model\predictions.csv')
simulation_model_pred = pd.read_csv('Leg_sim_model\leg_sim_predictions.csv')


combined_model_mse = mean_squared_error(avg_df['average_height'][40:],combined_model_pred.iloc[40:,-1])
experimental_model_mse = mean_squared_error(avg_df['average_height'][40:],experimental_model_pred.iloc[40:,-1])
simulation_model__mse = mean_squared_error(avg_df['average_height'][40:],simulation_model_pred.iloc[40:,-1])

print("wthout moving average")
print(f'cmb: {combined_model_mse}, exp: {experimental_model_mse}, sim: {simulation_model__mse}')

combined_model_mse = mean_squared_error(avg_df['smoothed_height'][40:],combined_model_pred.iloc[40:,-1])
experimental_model_mse = mean_squared_error(avg_df['smoothed_height'][40:],experimental_model_pred.iloc[40:,-1])
simulation_model__mse = mean_squared_error(avg_df['smoothed_height'][40:],simulation_model_pred.iloc[40:,-1])

print("with moving winow size: 3")
print(f'cmb: {combined_model_mse}, exp: {experimental_model_mse}, sim: {simulation_model__mse}')

combined_model_mse = mean_squared_error(avg_df['double_smoothed_height'][40:],combined_model_pred.iloc[40:,-1])
experimental_model_mse = mean_squared_error(avg_df['double_smoothed_height'][40:],experimental_model_pred.iloc[40:,-1])
simulation_model__mse = mean_squared_error(avg_df['double_smoothed_height'][40:],simulation_model_pred.iloc[40:,-1])

print("with moving winow size: 6")
print(f'cmb: {combined_model_mse}, exp: {experimental_model_mse}, sim: {simulation_model__mse}')

plt.scatter(avg_df['angle'], avg_df['smoothed_height'], label = 'experimental_data')
plt.plot(angles, combined_model_pred.iloc[:,-1], color = 'red', label = 'combined_model')
plt.plot(angles, experimental_model_pred.iloc[:,-1],color = 'orange', label = 'expreimental_model')
plt.plot(angles, simulation_model_pred.iloc[:,-1], color= 'yellow', label = 'simulation_model')
plt.title('predictions of 3 models')
plt.xlabel('angles')
plt.ylabel('height')
plt.legend()
plt.savefig('Plots/combined_plot')
plt.show()

from DNN import DNN
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

### Model
layers = [2, 128, 128, 128, 128, 128, 1]
leg_model = DNN(layers).to(device=device)

### data 
data= pd.read_excel("height_angle_data.xlsx")
sim_data = pd.read_csv('leg_sim_predictions.csv')
sim_data['4'] = sim_data['4'] - 3.1
map_dict = {}
for i in range(len(sim_data)):
    map_dict.update({sim_data.iloc[i,2]: sim_data.iloc[i,4]} )

data['sim_height']  = data['angle'].map(map_dict)

x= data[['angle', 'direction']].to_numpy()
y = data[['height', 'sim_height']].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train).to(device=device, dtype= torch.float32)
y_train = torch.tensor(y_train).to(device=device, dtype= torch.float32)

x_val = torch.tensor(x_val).to(device=device, dtype= torch.float32)
y_val = torch.tensor(y_val).to(device=device, dtype= torch.float32)

#test
angles = np.reshape(np.linspace(40, 180, 141), (141,1))
d_angles = angles = np.reshape(np.linspace(180, 40,141), (141,1))
d_angles= d_angles[::-1]

angles = np.concatenate((angles,d_angles), axis=0)
ups    = np.ones((141,1))
downs = np.zeros((141,1))
ups = np.concatenate((ups, downs) ,axis= 0)


x_test = np.concatenate((angles, ups), axis= 1 )
x_test = torch.tensor(x_test).to(device=device, dtype= torch.float32)# Model


train = True
load = False
num_epoch = 2500
learning_rate = 0.001
exp_weight = 0.9
sim_wieght = 0.1
weight_decay = 0.01


save_file = "Leg_combined_model\leg_model(sim+exp).pt"
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(leg_model.parameters(), lr= learning_rate)#, weight_decay= weight_decay)

loss_list = np.zeros((num_epoch, 3))

if __name__ == "__main__":
    # print(x_train.dtype)
    # print(leg_model(x_train))
    if load:
        leg_model.load_state_dict(torch.load(save_file))
    if train:
        for i in range(num_epoch):
            y_pred = leg_model(x_train)
            #l2_regularization = 0.5 * (leg_model.fc1.weight.norm(2)**2 + model.fc2.weight.norm(2)**2)
            loss = exp_weight* loss_fn(y_pred,torch.reshape(y_train[:,0], (len(y_train), 1)))
            
            loss += sim_wieght * loss_fn(y_pred,torch.reshape(y_train[:,1], (len(y_train), 1)))

            l2_regularization = 0.0
            for param in leg_model.parameters():
                l2_regularization += torch.norm(param, p=2)  # L2 norm of the weights

            loss += weight_decay * l2_regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_pred = leg_model(x_val)
            val_loss = loss_fn(val_pred, torch.reshape(y_val[:,0], (len(y_val), 1)))
            loss_list[i,:] = [i, loss.item(), val_loss.item()]
            print(f'Epoch:{i} loss: {loss} val_loss: {val_loss}')
    # x_test =   
    # pred = leg_model(x_test)
        torch.save(leg_model.state_dict(), save_file)
 
    y_test = leg_model(x_test)
    x_test = x_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    index = np.where(x == 1)
    x = x[index, 0]
    y = y[index,0 ]

    
    plt.scatter(x, y, label = 'experimental data')
    plt.plot(x_test[:,0], y_test, color ='red', label = 'model prediction')
    plt.plot(sim_data['2'],sim_data['4'], color ='green', label = 'simulation data' )
    plt.legend()
    plt.xlabel('angles')
    plt.ylabel('height')
    plt.title('Leg_model Data + Simulation')
    plt.savefig('Leg_combined_model/fig2')
    plt.show()

    pred = np.concatenate((x_test, y_test), axis= 1)
    pred = pd.DataFrame(pred)
    pred.to_csv('Leg_combined_model\predictions_up_down.csv', index=False)

    # plt.plot(loss_list[:,0], loss_list[:,1], color = 'blue', label = 'train_loss')
    # plt.plot(loss_list[:,0], loss_list[:,2], color = 'red', label = 'valid_loss')
    # plt.title('Loss combine model')
    # plt.legend()
    # plt.savefig('Leg_combined_model/convergance_plot')
    # plt.show()
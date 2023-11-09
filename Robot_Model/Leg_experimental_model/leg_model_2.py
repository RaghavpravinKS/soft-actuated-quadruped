from DNN import DNN
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


save_file = "leg_model(exp).pt"

### Model
layers = [2, 128, 128, 128, 128, 128, 1]
leg_model = DNN(layers).to(device=device)

#data 
data= pd.read_excel("height_angle_data.xlsx")

x= data[['angle', 'direction']].to_numpy()
y = data['height'].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

y_train = np.reshape(y_train, (len(y_train), 1))
y_val = np.reshape(y_val, (len(y_val), 1))

x_train = torch.tensor(x_train).to(device=device, dtype= torch.float32)
y_train = torch.tensor(y_train).to(device=device, dtype= torch.float32)

x_val = torch.tensor(x_val).to(device=device, dtype= torch.float32)
y_val = torch.tensor(y_val).to(device=device, dtype= torch.float32)

#test
angles = np.reshape(np.linspace(40, 180, 180), (180,1))
d_angles = angles = np.reshape(np.linspace(180, 40,180), (180,1))
d_angles= d_angles[::-1]

angles = np.concatenate((angles,d_angles), axis=0)
ups    = np.ones((180,1))
downs = np.zeros((180,1))

ups = np.concatenate((ups, downs) ,axis= 0)

x_test = np.concatenate((angles, ups), axis= 1 )
x_test = torch.tensor(x_test).to(device=device, dtype= torch.float32)# Model


train = True
load = True
num_epoch = 2000
learning_rate = 0.001
weight_decay = 0.005 

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(leg_model.parameters(), lr= 0.001, weight_decay= weight_decay)

if __name__ == "__main__":
    # print(x_train.dtype)
    # print(leg_model(x_train))
    if load:
        leg_model.load_state_dict(torch.load(save_file))
    if train:
        for i in range(num_epoch):
            y_pred = leg_model(x_train)
            #l2_regularization = 0.5 * (leg_model.fc1.weight.norm(2)**2 + model.fc2.weight.norm(2)**2)
            loss = loss_fn(y_pred, y_train)

            l2_regularization = 0.0
            for param in leg_model.parameters():
                l2_regularization += torch.norm(param, p=2)  # L2 norm of the weights

            loss += weight_decay * l2_regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_pred = leg_model(x_val)
            val_loss = loss_fn(val_pred, y_val)
            print(f'Epoch:{i} loss: {loss} val_loss: {val_loss}')
    # x_test =   
    # pred = leg_model(x_test)
        torch.save(leg_model.state_dict(), save_file)

    y_test = leg_model(x_test)
    x_test = x_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    plt.scatter(x[:,0], y)
    plt.plot(x_test[:,0], y_test, color ='red')
    plt.xlabel('angles')
    plt.ylabel('height')
    plt.title('Data vs Prediction')
    plt.legend()
    plt.show()
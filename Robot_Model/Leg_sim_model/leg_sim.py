from DNN import DNN
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

layers = [3, 128, 128, 128, 128, 128,  2]
leg_model = DNN(layers).to(device=device)


filename = "Leg_sim_model\leg_data.csv"
save_file = "Leg_sim_model\leg_model.pt"

data= pd.read_csv(filename)
data.iloc[:, 4] =  data.iloc[:, 4] / 3.14 * 480  - 70
data.iloc[:, 6] =  data.iloc[:, 6]  * 100 

x_train = data.iloc[:, [1,2,4]].to_numpy()   # Fx,Fy, alpha
y_train = data.iloc[:, [5,6]].to_numpy()     # x, y

# min_values = np.min(x_train, axis=0, keepdims=True)
# max_values = np.max(x_train, axis=0, keepdims=True)
# x_train = (x_train - min_values) / (max_values - min_values)

x_train = torch.tensor(x_train).to(device=device, dtype= torch.float32)
y_train = torch.tensor(y_train).to(device=device, dtype= torch.float32)


## Validation
val_data= pd.read_csv("leg_val_data.csv")
val_data.iloc[:, 4] =  val_data.iloc[:, 4] / 3.14 * 480  - 70
val_data.iloc[:, 6] =  val_data.iloc[:, 6]  * 100 

x_val = val_data.iloc[:, [1,2,4]].to_numpy()   # Fx,Fy, alpha
y_val= val_data.iloc[:, [5,6]].to_numpy()     # x, y

# x_val= (x_val - min_values) / (max_values - min_values)

x_val = torch.tensor(x_val).to(device=device, dtype= torch.float32)
y_val = torch.tensor(y_val).to(device=device, dtype= torch.float32)

angles = np.reshape(np.linspace(40, 180, 141), (141,1))
Fx     = np.zeros((141,1))
Fy     = -1.5 * np.ones((141,1))
x_test = np.concatenate((Fx, Fy, angles), axis=1)
x_test = torch.tensor(x_test).to(device=device, dtype= torch.float32)


train = False
load = True
num_epoch = 2500
learning_rate = 0.001
weight_decay = 0.005 

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(leg_model.parameters(), lr= learning_rate, weight_decay= weight_decay)

loss_list = np.zeros((num_epoch, 3))
if __name__ == "__main__":
    # print(x_train.dtype)
    # print(leg_model(x_train))
    if load:
        leg_model.load_state_dict(torch.load(save_file))
    if train:
        for i in range(num_epoch):
            y_pred = leg_model(x_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            l2_regularization = 0.0
            for param in leg_model.parameters():
                l2_regularization += torch.norm(param, p=2)  # L2 norm of the weights

            loss += weight_decay * l2_regularization
            val_pred = leg_model(x_val)
            val_loss = loss_fn(val_pred, y_val)
            loss_list[i,:] = [i, loss.item(), val_loss.item()]
            print(f'Epoch:{i} loss: {loss} val_loss: {val_loss}')
       
    
        torch.save(leg_model.state_dict(), save_file)
    
    y_test = leg_model(x_test)
    x_test = x_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
  
    
    plt.plot(x_test[:,2], y_test[:, 1], color ='red', label = 'prediction')
    plt.xlabel('angles')
    plt.ylabel('height')
    plt.title('Simulation Model')   
    plt.legend()
    plt.savefig('Leg_sim_model/figure1')
    plt.show()

    #print(loss_list)
    # plt.plot(loss_list[:,0], loss_list[:,1], color = 'blue', label = 'train_loss')
    # plt.plot(loss_list[:,0], loss_list[:,2], color = 'red', label = 'valid_loss')
    # plt.title('Loss simulation model')
    # plt.legend()
    # plt.savefig('Leg_sim_model/convergance_plot')
    # plt.show()

    pred = np.concatenate((x_test, y_test), axis= 1)
    pred = pd.DataFrame(pred)
    pred.to_csv('leg_sim_predictions.csv', index=False)


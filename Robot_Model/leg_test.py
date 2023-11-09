
from DNN import DNN
import torch 
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)

leg_layers = [3, 128, 128, 128, 128, 128,  2]
leg_model = DNN(leg_layers).to(device=device)
leg_model.load_state_dict(torch.load("leg_model.pt"))


if __name__ == '__main__':
    Fx = np.zeros((180,1))
    Fy = -1.5 * np.ones((180,1))

    angels = np.reshape(range(1,181), (180,1))
    input  = np.concatenate((Fx,Fy, angels), axis = 1   )
    input = torch.tensor(input).to(device = device, dtype=torch.float32)
    pred = leg_model(input)
    pred = pred.detach().cpu().numpy()

    plt.plot(angels, pred[:,1])

    Fx = - 0.5 * np.ones((180,1))
    Fy = -1 * np.ones((180,1))

    angels = np.reshape(range(1,181), (180,1))
    input  = np.concatenate((Fx,Fy, angels), axis = 1   )
    input  = torch.tensor(input).to(device = device, dtype=torch.float32)
    pred   = leg_model(input)
    pred   = pred.detach().cpu().numpy()

    plt.plot(angels, pred[:,1])
    
    plt.show()
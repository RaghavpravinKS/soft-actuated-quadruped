import socket
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

"""#########################################"""
"""######### Setting up connection #########"""
"""#########################################"""

#IP address, port for the socket server
host = '192.168.168.239'
port = 1234

#socket server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("Socket server listening on {}:{}".format(host, port))

client_socket, addr = server_socket.accept()
print("Connected to ESP32:", addr)

global init_wall_dist
init_wall_dist= 80

def datRead():
    data = client_socket.recv(1024).decode().strip()
    return data.split()

def datSend():
    response=""
    client_socket.sendall(response.encode())

def resetBot():
    response="RESET\n"
    client_socket.sendall(response.encode())
    data = client_socket.recv(1024).decode().strip()
    while(data!="done"):
        data = client_socket.recv(1024).decode().strip()
        pass
    print("reset successful")

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        #State space degfined by the sensor readings
        self.imu_dim = 3
        self.usonic_dim = 3
        self.fsr_dim = 4
        self.state_dim = self.imu_dim + self.usonic_dim + self.fsr_dim
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        #Action space for the 12 servos
        self.action_dim = 12
        self.action_space = spaces.Box(low=0, high=180, shape=(self.action_dim,), dtype=np.float32)

        #Current state variable
        self.current_state = None

         #Timer initialization
        self.start_time = None
        self.last_time = None

    def reset(self):
        self.current_state = np.zeros(self.state_dim)
        self.start_time = time.time()
        self.last_time=0
        return self.current_state

    def step(self, action):
        dat=datRead();
        imu_data = [dat[0], dat[1], dat[2]]
        usonic_data = [dat[3], dat[4], dat[5]]
        fsr_data = [dat[6], dat[7], dat[8], dat[9]]
        new_state = np.concatenate((imu_data, usonic_data, fsr_data))

        elapsed_time = time.time() - self.start_time
        speed= usonic_data[0]/(elapsed_time-self.last_time)

        reward = speed

        self.last_time= elapsed_time

        return new_state, reward, {}

env = CustomEnv()
state = env.reset()
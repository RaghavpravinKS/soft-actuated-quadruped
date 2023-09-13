import socket

# IP address and port for the socket server
host = '192.168.168.239'
port = 1234

# Set up the socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("Socket server listening on {}:{}".format(host, port))

# Accept incoming connections
client_socket, addr = server_socket.accept()
print("Connected to ESP32:", addr)

def sendReset():
    response="RESET\n"
    client_socket.sendall(response.encode())
    data = client_socket.recv(1024).decode().strip()
    while(data!="done"):
        data = client_socket.recv(1024).decode().strip()
        pass
    print("reset successful")


while True:
    # Receive sensor data from the ESP32
    data = client_socket.recv(1024).decode().strip()
    if data:
        print(data.split())
        # Process the received data or perform desired actions

    # You can also send a response to the ESP32 if needed
    # response = "Response message"
    # client_socket.send(response.encode())
    # sendReset()


# Close the socket connection
client_socket.close()
server_socket.close()

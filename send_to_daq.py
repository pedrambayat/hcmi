import socket
from smile_detector import realtime_smile_detector
result = realtime_smile_detector()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('localhost', 51001))
    s.listen(1)
    print('run MATLAB file')
    conn, addr = s.accept()  
    print(f"Connected by {addr}")
    conn.sendall(bytes(str(result), 'ASCII'))
    conn.close()
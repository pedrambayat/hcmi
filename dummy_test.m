u = udpport('LocalPort', 3000);

while u.NumBytesAvailable == 0
    pause(1);
end

data = read(u, u.NumBytesAvailable);
message = char(data');

disp(message);

clear u
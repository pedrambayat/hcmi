function judgement(dq, n, f_ext, f_flex, A_ext, A_flex)
    u = udpport('LocalPort', 3000);
    
    while u.NumBytesAvailable == 0
        pause(1);
    end
    
    data = read(u, u.NumBytesAvailable);
    message = uint8(data);
    
    if message == 1
        disp('Judge APPROVES');
        out_roach = A_ext*sin(linspace(0, 2*pi*f_ext,n))';
        out_green = ones(n, 1); % green LEDs
        out_red = zeros(n, 1); % red LEDs
    
    elseif message == 0
        disp('Judge DOES NOT APPROVE');
        out_roach = A_flex*sin(linspace(0, 2*pi*f_flex,dq.Rate)');
        out_green = zeros(n, 1); % green LEDs
        out_red = ones(n, 1); % red LEDs
    end

    write(dq, [out_roach, out_green, out_red]);
    
    clear u
end
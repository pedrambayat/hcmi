function judge_dance() 
    %%% SET UP DATA ACQUISITION %%%
    % add device instance
    dq = daq('ni');
    dq.Rate = 6000; % Hz
    n = dq.Rate * 5;

    addoutput(dq, 'myDAQ1', 'ao0', 'Voltage');

    % cockroach leg variables 
    f_ext = 2000;
    f_flex = 200;
    A_ext = 0.5;
    A_flex = 1;

    % TCP read
    t = tcpclient('localhost', 51001);
    
    % allow for buffer to recieve data
    pause(0.05);

    num = t.NumBytesAvailable;
    if num > 0
        output = read(t, num, 'string');  % read python output as text
        data = strtrim(output);  
    else
        data = '';
    end

    if data == "Approve"
        disp('Judge approves!');
        out_roach = A_ext * sin(linspace(0, 2*pi*f_ext, n))';
    
    elseif data == "Disapprove"
        disp('Judge does not approve :(');
        out_roach = A_flex * sin(linspace(0, 2*pi*f_flex, n))';
    elseif data == "Tie"
        disp('Judge thinks it is a tie!');
        out_roach = zeros(n, 1);
    else
        disp('No judge found.');
        out_roach = zeros(n,1);
    end
    
    % output signal to roach leg
    write(dq, out_roach);

    while dq.Running
        pause(1);
    end
    
    % terminate TCP, dq objects from workspace!
    clear t
    clear dq
end

function judgement_FINAL() 
    dq = daq('ni');
    dq.Rate = 6000;
    n = dq.Rate * 5;

    addoutput(dq, 'myDAQ1', 'ao0', 'Voltage');

    % cockroach leg variables
    f_ext = 1000;
    f_flex = 250;
    A_ext = 0.5;
    A_flex = 0.5;

    % TCP read
    t = tcpclient('localhost', 51001);

    pause(0.05);

    num = t.NumBytesAvailable;
    if num > 0
        output = read(t, num, "string");  % read as text
        data = strtrim(output);  
    else
        data = "";
    end

    if data == "Approve"
        disp('Judge APPROVES');
        out_roach = A_ext * sin(linspace(0, 2*pi*f_ext, n))';
    
    elseif data == "Disapprove"
        disp('Judge DOES NOT APPROVE');
        out_roach = A_flex * sin(linspace(0, 2*pi*f_flex, n))';
    elseif data == "Tie"
        disp('Judge thinks it is a tie!');
        out_roach = zeros(n, 1);
    else
        disp('No judge found.');
        out_roach = zeros(n,1);
    end

    write(dq, out_roach);
    clear t
end

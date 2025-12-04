function calibrate_roach_leg()
    %%% SET UP DATA ACQUISITION %%%
    % add device instance
    dq = daq('ni');
    dq.Rate = 6000; % Hz
    n = dq.Rate * 5;

    addoutput(dq, 'myDAQ1', 'ao0', 'Voltage');

    % cockroach leg variables
    % f_ext = 2000;
    % A_ext = 0.5;
    % out_roach = A_ext * sin(linspace(0, 2*pi*f_ext, n))';

    f_flex = 200;
    A_flex = 1;
    out_roach = A_flex * sin(linspace(0, 2*pi*f_flex, n))';
    
    % output signal to roach leg
    write(dq, out_roach);

    while dq.Running
        disp('Signal out.')
    end

    % % clear dq from workspace!
    % clear dq
end
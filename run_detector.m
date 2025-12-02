clear; clc; close all;

verdict = realtime_smile_detector();

fprintf('Final verdict: %s\n', verdict);

dq = daq('ni');
dq.Rate = 6000; % max f for roach = 5KHz
n = dq.Rate * 5; % i.e. duration of signal = 5s

addoutput(dq, 'myDAQ1', 'ao0', 'Voltage'); % cockroach leg
addoutput(dq, 'myDAQ1', 'port0/line0', 'Digital'); % green LEDs
addoutput(dq, 'myDAQ1', 'port0/line1', 'Digital'); % red LEDs

% move leg based on judge
if strcmp(verdict, 'Approve')
    disp('Dancer approved!');
    out_roach = A_ext*sin(linspace(0, 2*pi*f_ext,n))';
    out_green = ones(n, 1); % green LEDs
    out_red = zeros(n, 1); % red LEDs    
    
elseif strcmp(verdict, 'Disapprove')
    disp('Dancer disapproved!');
    out_roach = A_flex*sin(linspace(0, 2*pi*f_flex,dq.Rate)');
    out_green = zeros(n, 1); % green LEDs
    out_red = ones(n, 1); % red LEDs
end

write(dq, [out_roach, out_green, out_red]);
disp('sent signal to roach leg');
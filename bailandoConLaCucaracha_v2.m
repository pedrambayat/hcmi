%%% INITIALIZE SERVO VARIABLES %%%
% step sizes for servo motor
fast_step = 0.01;
med_step = 0.0075;
slow_step = 0.003;

% ROM of servo motor
pwm_start = 0.1;
pwm_stop = 0.05; % CW from 0.1

% emg threhsolds
upper_thresh = 0.04;
lower_thresh = 0.01; 

%%% SET UP DATA ACQUISITION %%%
% add device instance
dq = daq('ni');
dq.Rate = 2000;
dq.ScansAvailableFcnCount = 250; % 8 readings per second

% initialize input channels
addinput(dq, 'myDAQ1', 'ai0', 'Voltage'); % emg

% initialize output channels
pwm = addoutput(dq,'myDAQ1', 'ctr0', 'PulseGeneration');

% set up PWM signal
pwm.Frequency = 50;
pwm.InitialDelay = 0;
pwm.DutyCycle = 0.1;

%%% CALIBRATE %%%
dq.ScansAvailableFcn = @(src,evt) calibration(src, evt);
start(dq, 'Duration', seconds(20));

%%%  START DANCING! %%%
% set up data acquisition
dq.ScansAvailableFcn = @(src,evt) move_servo(src, evt, fast_step, med_step, slow_step, pwm_start, pwm_stop, upper_thresh, lower_thresh);

% load song
[x,Fs] = audioread('laCucaracha_trimmed.mp3');

% play song in the background, stop data acquisition when song ends
p = audioplayer(x, Fs);
p.StopFcn = @(src,evt) stop(dq);

% start acquiring data from the emg
start(dq, 'Continuous');
play(p); 

%%% INITIALIZE ROACH/LED VARIABLES %%%
% cockroach leg
f_ext = 1000;
f_flex = 250;
A_ext = 0.5;
A_flex = 0.5;

dq.Rate = 6000; % max f for roach = 5KHz
n = dq.Rate * 5; % i.e. duration of signal = 5s

% initialize output channels
addoutput(dq, 'myDAQ1', 'ao0', 'Voltage'); % cockroach leg
addoutput(dq, 'myDAQ1', 'port0/line0', 'Digital'); % green LEDs
addoutput(dq, 'myDAQ1', 'port0/line1', 'Digital'); % red LEDs

judgement(dq, n, f_ext, f_flex, A_ext, A_flex)
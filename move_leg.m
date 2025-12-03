function move_leg(dq, n, f_ext, f_flex, A_ext, A_flex)
    clear; clc; close all;
    
    verdict = realtime_smile_detector();
    
    fprintf('Final verdict: %s\n', verdict);
    
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
end
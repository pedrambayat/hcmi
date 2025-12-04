function move_servo(src, ~, fast_step, med_step, slow_step, pwm_start, pwm_stop, upper_thresh, lower_thresh)
    %%% LOCAL VARS %%%
    persistent dir pwm
    if isempty(dir)
        % start servo motion CW toward pwm_stop
        dir = -1; 

        % set up PWM signal
        % pwm = src.Channels(2); % need to change after adding other channels
    end

    pwm = src.Channels(strcmp({src.Channels.ID}, 'ctr0'));
    
    %%% COMPUTE INTEGRAL %%%
    % read in data
    [raw_data, timestamps, ~] = read(src, src.ScansAvailableFcnCount, 'OutputFormat', 'Matrix');

    % make sure it is centered at 0
    avg = mean(raw_data);
    clean_data = raw_data - avg;

    integral = trapz(timestamps, abs(clean_data));

    % assign step size
    if integral > upper_thresh
        step_size = fast_step;
        disp('STRONG FLEX');
    elseif integral < upper_thresh && integral > lower_thresh
        step_size = med_step;
        disp('WEAK FLEX');
    else
        step_size = slow_step;
        disp('NO FLEX');
    end
    
    % clamp duty cycle
    % (max, min) = (pwm_start, pwm_stop)
    duty_0 = pwm.DutyCycle;
    duty_1 = duty_0 + dir * step_size;

    if duty_1 > pwm_start
        duty_1 = pwm_start;
        dir = -1;
    end
    
    if duty_1 < pwm_stop
        duty_1 = pwm_stop;
        dir = 1;
    end

    pwm.DutyCycle = duty_1;
end
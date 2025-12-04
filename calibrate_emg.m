function calibrate_emg(src, ~)
    % read in data
    [raw_data, timestamps, ~] = read(src, src.ScansAvailableFcnCount, 'OutputFormat', 'Matrix');

    % make sure data is centered at 0
    avg = mean(raw_data);
    clean_data = raw_data - avg;
    
    % compute integral
    integral = trapz(timestamps, abs(clean_data));
    
    % display results
    % plot(timestamps, integral);
    fprintf('Integral = %.3f\n', integral);
end
clear; clc; close all;

verdict = realtime_smile_detector();

fprintf('Final verdict: %s\n', verdict);

if strcmp(verdict, 'Approve')
    % Do something for approval
    disp('Person approved!');
elseif strcmp(verdict, 'Disapprove')
    % Do something for disapproval
    disp('Person disapproved!');
end

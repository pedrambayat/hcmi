%% Quick Start Script for Real-Time Smile Detection
clear; clc; close all;

% Make sure you're in the directory with the Python script and model files
fprintf('Current directory: %s\n', pwd);
fprintf('Make sure smile_frown_resnet18.pth and pytorch_matlab_wrapper.py are here!\n\n');

% Run the detector
realtime_smile_detector();
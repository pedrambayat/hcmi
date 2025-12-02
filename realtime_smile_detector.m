function realtime_smile_detector()
    % Setup Python environment
    fprintf('Setting up Python environment...\n');
    pe = pyenv;
    if pe.Status == "NotLoaded"
        pyenv('Version', '/Library/Developer/CommandLineTools/usr/bin/python3');  % Or specify full path to your python
    end
    
    % Add current directory to Python path
    if count(py.sys.path, pwd) == 0
        insert(py.sys.path, int32(0), pwd);
    end
    
    % Initialize PyTorch model
    fprintf('Loading PyTorch model...\n');
    try
        py.pytorch_matlab_wrapper.initialize('smile_frown_resnet18.pth', 'class_mapping.json');
        fprintf('Model loaded successfully!\n\n');
    catch ME
        error('Failed to load model: %s', ME.message);
    end
    
    %%Setup webcam
    fprintf('Initializing webcam...\n');
    cam = webcam;
    fprintf('Webcam initialized: %s\n', cam.Name);

    % Get a test frame to determine actual resolution
    testFrame = snapshot(cam);
    [frameHeight, frameWidth, ~] = size(testFrame);
    fprintf('Camera resolution: %d x %d\n', frameWidth, frameHeight);    
    
    % Setup face detector
    faceDetector = vision.CascadeObjectDetector('MinSize', [100 100]);
    
    %% Setup DAQ (uncomment and modify for your hardware)
    % fprintf('Setting up DAQ...\n');
    % daqDevice = daq("ni");
    % addoutput(daqDevice, "Dev1", "ao0", "Voltage");
    % fprintf('DAQ ready\n\n');
    
    % Setup figure for display
    hFig = figure('Name', 'Real-Time Smile/Frown Detector', ...
                  'NumberTitle', 'off', ...
                  'WindowState', 'maximized', ...
                  'CloseRequestFcn', @closeFigure);
    
    % Create axes for video display with proper aspect ratio
    hAx = axes('Parent', hFig, 'Position', [0.05 0.15 0.9 0.8]);
    axis(hAx, 'image');  % This maintains aspect ratio
    hImg = imshow(testFrame, 'Parent', hAx);  % Use actual frame size
    set(hAx, 'Units', 'normalized');
    
    % Create text display for stats
    hText = annotation('textbox', [0.05 0.01 0.9 0.1], ...
                      'String', 'Initializing...', ...
                      'FontSize', 14, ...
                      'EdgeColor', 'none', ...
                      'HorizontalAlignment', 'center', ...
                      'FontWeight', 'bold');
    
    % Performance tracking
    frameCount = 0;
    startTime = tic;
    fpsWindow = zeros(30, 1);  % Rolling window for FPS calculation
    fpsIdx = 1;
    
    % Main loop
    fprintf('Starting real-time detection...\n');
    fprintf('Press Ctrl+C or close the figure to stop.\n\n');
    fprintf('%-10s | %-10s | %-12s | %-8s | %-15s\n', ...
            'Frame', 'FPS', 'Prediction', 'Conf.', 'Probs [F, S]');
    fprintf('%s\n', repmat('-', 1, 75));
    
    stopFlag = false;
    
    while ishandle(hFig) && ~stopFlag
        frameStart = tic;
        
        try
            % Capture frame from webcam
            frame = snapshot(cam);
            
            % Detect faces
            bboxes = faceDetector(frame);
            
            predictionText = 'No face detected';
            daqSignal = 0;  % Neutral signal when no face
            
            if ~isempty(bboxes)
                % Use the largest face (first one is usually largest)
                bbox = bboxes(1, :);
                
                % Expand bbox slightly for better context (optional)
                expandRatio = 0.1;
                bbox = expandBBox(bbox, size(frame), expandRatio);
                
                % Extract face region
                face = imcrop(frame, bbox);
                
                % Ensure RGB uint8 format
                if size(face, 3) == 1
                    face = repmat(face, [1, 1, 3]);
                end
                face = uint8(face);
                
                % Run PyTorch inference
                np_face = py.numpy.array(face);
                result = py.pytorch_matlab_wrapper.predict(np_face);
                
                % Extract results
                label = char(result{'label'});
                confidence = double(result{'confidence'});
                probs = double(result{'probabilities'});
                
                % Prepare display text
                predictionText = sprintf('%s: %.1f%%\n[Frown: %.2f | Smile: %.2f]', ...
                                       upper(label), confidence*100, probs(1), probs(2));
                
                % Determine DAQ signal based on prediction
                if strcmp(label, 'smile')
                    daqSignal = 5.0;  % High signal for smile
                    boxColor = [0 255 0];  % Green
                else
                    daqSignal = 0.5;  % Low signal for frown
                    boxColor = [255 0 0];  % Red
                end
                
                % Send signal to DAQ
                % write(daqDevice, daqSignal);
                
                % Draw bounding box and label on frame
                frame = insertShape(frame, 'Rectangle', bbox, ...
                                   'Color', boxColor, 'LineWidth', 3);
                frame = insertText(frame, [bbox(1), bbox(2)-10], ...
                                  sprintf('%s %.1f%%', upper(label), confidence*100), ...
                                  'FontSize', 18, 'BoxColor', boxColor, ...
                                  'BoxOpacity', 0.7, 'TextColor', 'white');
                
                % Console output (every 10 frames to avoid clutter)
                if mod(frameCount, 10) == 0
                    fprintf('%-10d | %-10.1f | %-12s | %-8.2f | [%.2f, %.2f]\n', ...
                            frameCount, 1/mean(fpsWindow), label, confidence, probs(1), probs(2));
                end
            else
                % No face detected - send neutral signal
                % write(daqDevice, daqSignal);
            end
            
            % Update display
            set(hImg, 'CData', frame);
            
            % Calculate FPS
            frameTime = toc(frameStart);
            fpsWindow(fpsIdx) = frameTime;
            fpsIdx = mod(fpsIdx, 30) + 1;
            avgFPS = 1 / mean(fpsWindow);
            
            % Update stats text
            elapsedTime = toc(startTime);
            statsText = sprintf(['Frame: %d | FPS: %.1f | Elapsed: %.1fs\n' ...
                               'DAQ Signal: %.2fV\n\n%s'], ...
                               frameCount, avgFPS, elapsedTime, daqSignal, predictionText);
            set(hText, 'String', statsText);
            
            drawnow limitrate;  % Limit redraw rate for performance
            
            frameCount = frameCount + 1;
            
        catch ME
            fprintf('Error in frame processing: %s\n', ME.message);
            stopFlag = true;
        end
    end
    
    % Cleanup
    fprintf('\n\nStopping detection...\n');
    clear cam;
    % if exist('daqDevice', 'var')
    %     stop(daqDevice);
    %     clear daqDevice;
    % end
    if ishandle(hFig)
        close(hFig);
    end
    fprintf('Cleanup complete. Total frames processed: %d\n', frameCount);
    
    % Nested function for figure close callback
    function closeFigure(~, ~)
        stopFlag = true;
        delete(hFig);
    end
end

function expandedBBox = expandBBox(bbox, frameSize, ratio)
    % Expand bounding box by ratio while staying within frame bounds
    x = bbox(1);
    y = bbox(2);
    w = bbox(3);
    h = bbox(4);
    
    % Calculate expansion
    dx = w * ratio;
    dy = h * ratio;
    
    % Apply expansion
    newX = max(1, x - dx);
    newY = max(1, y - dy);
    newW = min(frameSize(2) - newX, w + 2*dx);
    newH = min(frameSize(1) - newY, h + 2*dy);
    
    expandedBBox = [newX, newY, newW, newH];
end
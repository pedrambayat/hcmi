function result = realtime_smile_detector()
    % Returns: 'Approve' if majority smiling, 'Disapprove' if majority frowning
    
    % Setup Python environment
    fprintf('Setting up Python environment...\n');
    pe = pyenv;
    if pe.Status == "NotLoaded"
        pyenv('Version', '/Library/Developer/CommandLineTools/usr/bin/python3');
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
    
    %% Setup webcam
    fprintf('Initializing webcam...\n');
    cam = webcam;
    fprintf('Webcam initialized: %s\n', cam.Name);

    % Get a test frame to determine actual resolution
    testFrame = snapshot(cam);
    [frameHeight, frameWidth, ~] = size(testFrame);
    fprintf('Camera resolution: %d x %d\n', frameWidth, frameHeight);    
    
    % Setup face detector
    faceDetector = vision.CascadeObjectDetector('MinSize', [100 100]);
    
    % Setup figure for display
    hFig = figure('Name', 'Real-Time Smile/Frown Detector - Press SPACE or close window to finish', ...
                  'NumberTitle', 'off', ...
                  'WindowState', 'maximized', ...
                  'KeyPressFcn', @keyPressCallback, ...
                  'CloseRequestFcn', @closeFigure);
    
    % Create axes for video display with proper aspect ratio
    hAx = axes('Parent', hFig, 'Position', [0.05 0.25 0.9 0.7]);
    axis(hAx, 'image');
    hImg = imshow(testFrame, 'Parent', hAx);
    set(hAx, 'Units', 'normalized');
    
    % Create text display for stats
    hText = annotation('textbox', [0.05 0.01 0.9 0.2], ...
                      'String', 'Initializing...', ...
                      'FontSize', 14, ...
                      'EdgeColor', 'none', ...
                      'HorizontalAlignment', 'center', ...
                      'FontWeight', 'bold');
    
    % Arrays to store predictions
    predictionResults = {};  % Cell array to store all predictions
    captureInterval = 1.0;   % Capture every 1 second
    lastCaptureTime = tic;
    
    % Performance tracking
    frameCount = 0;
    startTime = tic;
    captureCount = 0;
    
    % Main loop
    fprintf('\n=== Starting Smile/Frown Assessment ===\n');
    fprintf('Recording one sample per second...\n');
    fprintf('Press SPACEBAR or close window when done.\n\n');
    fprintf('%-10s | %-12s | %-10s | %-15s\n', ...
            'Sample #', 'Prediction', 'Conf.', 'Running Count');
    fprintf('%s\n', repmat('-', 1, 60));
    
    stopFlag = false;
    
    while ishandle(hFig) && ~stopFlag
        try
            % Capture frame from webcam
            frame = snapshot(cam);
            
            % Detect faces
            bboxes = faceDetector(frame);
            
            predictionText = 'Waiting for face...';
            currentPrediction = '';
            
            % Check if it's time to capture and analyze
            shouldCapture = toc(lastCaptureTime) >= captureInterval;
            
            if ~isempty(bboxes) && shouldCapture
                % Use the largest face
                bbox = bboxes(1, :);
                
                % Expand bbox slightly
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
                result_dict = py.pytorch_matlab_wrapper.predict(np_face);
                
                % Extract results
                label = char(result_dict{'label'});
                confidence = double(result_dict{'confidence'});
                probs = double(result_dict{'probabilities'});
                
                % Store the prediction
                predictionResults{end+1} = struct('label', label, ...
                                                  'confidence', confidence, ...
                                                  'timestamp', toc(startTime));
                captureCount = captureCount + 1;
                
                % Count current results
                smileCount = sum(cellfun(@(x) strcmp(x.label, 'smile'), predictionResults));
                frownCount = sum(cellfun(@(x) strcmp(x.label, 'frown'), predictionResults));
                
                % Console output
                fprintf('%-10d | %-12s | %-10.2f | Smile: %d, Frown: %d\n', ...
                        captureCount, upper(label), confidence, smileCount, frownCount);
                
                currentPrediction = label;
                
                % Reset capture timer
                lastCaptureTime = tic;
            end
            
            % Visualize current frame
            if ~isempty(bboxes)
                bbox = bboxes(1, :);
                expandRatio = 0.1;
                bbox = expandBBox(bbox, size(frame), expandRatio);
                
                % Color based on last capture if available
                if ~isempty(currentPrediction)
                    if strcmp(currentPrediction, 'smile')
                        boxColor = [0 255 0];  % Green
                    else
                        boxColor = [255 0 0];  % Red
                    end
                else
                    boxColor = [255 255 0];  % Yellow (waiting)
                end
                
                frame = insertShape(frame, 'Rectangle', bbox, ...
                                   'Color', boxColor, 'LineWidth', 3);
            end
            
            % Update display
            set(hImg, 'CData', frame);
            
            % Calculate current counts
            smileCount = sum(cellfun(@(x) strcmp(x.label, 'smile'), predictionResults));
            frownCount = sum(cellfun(@(x) strcmp(x.label, 'frown'), predictionResults));
            
            % Time until next capture
            timeToNext = max(0, captureInterval - toc(lastCaptureTime));
            
            % Update stats text
            elapsedTime = toc(startTime);
            statsText = sprintf(['Samples Captured: %d | Elapsed: %.1fs | Next capture in: %.1fs\n\n' ...
                               'SMILE: %d samples (%.1f%%)\n' ...
                               'FROWN: %d samples (%.1f%%)\n\n' ...
                               'Press SPACEBAR or close window to finish'], ...
                               captureCount, elapsedTime, timeToNext, ...
                               smileCount, (smileCount/(captureCount+eps))*100, ...
                               frownCount, (frownCount/(captureCount+eps))*100);
            set(hText, 'String', statsText);
            
            drawnow limitrate;
            frameCount = frameCount + 1;
            
        catch ME
            fprintf('Error in frame processing: %s\n', ME.message);
            stopFlag = true;
        end
    end
    
    % Cleanup
    fprintf('\n\nAssessment Complete\n');
    clear cam;
    if ishandle(hFig)
        close(hFig);
    end
    
    % Analyze results
    if isempty(predictionResults)
        fprintf('No samples captured. Cannot make determination.\n');
        result = 'Insufficient Data';
        return;
    end
    
    % Count smiles and frowns
    smileCount = sum(cellfun(@(x) strcmp(x.label, 'smile'), predictionResults));
    frownCount = sum(cellfun(@(x) strcmp(x.label, 'frown'), predictionResults));
    totalSamples = length(predictionResults);
    
    fprintf('\nTotal samples: %d\n', totalSamples);
    fprintf('Smiles: %d (%.1f%%)\n', smileCount, (smileCount/totalSamples)*100);
    fprintf('Frowns: %d (%.1f%%)\n', frownCount, (frownCount/totalSamples)*100);
    
    % Determine final result
    if smileCount > frownCount
        result = 'Approve';
        fprintf('\n*** RESULT: APPROVE ***\n');
    elseif frownCount > smileCount
        result = 'Disapprove';
        fprintf('\n*** RESULT: DISAPPROVE ***\n');
    else
        result = 'Tie';
        fprintf('\n*** RESULT: TIE (Equal smiles and frowns) ***\n');
    end
    
    fprintf('\nReturning: %s\n', result);
    
    % Nested functions
    function closeFigure(~, ~)
        stopFlag = true;
    end
    
    function keyPressCallback(~, event)
        if strcmp(event.Key, 'space')
            stopFlag = true;
        end
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
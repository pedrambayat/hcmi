# Human-Cockroach Machine Interface: Facial Expression Classification Model

The following MATLAB packages are required:
- Computer Vision Toolbox
- Image Processing Toolbox
- MATLAB Support Package for USB Webcams
- Data Acquisition Toolbox

The following python packages are required:
`python -m pip install --user torch torchvision pillow numpy`

Then, in `realtime_smile_detector.m`, update `pyenv` to the python path and update the `.pth` file to the correct model.

Finally, call `run_detector` in the MATLAB command window.

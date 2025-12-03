# Dual-Platform Facial Expression Inference for Cockroach-Machine Interface

Training dataset should be structured as:
`dataset\ 
    \dataset_{name}
        \train
            \smile
            \frown
        \test
            \smile
            \frown
`   
To train, record a video of a judge smiling and a video of a judge frowning.
Then, use ffmpeg to extract images. This will quickly create many images that can be used for training.

`ffmpeg -i input.mp4 -vf "select=not(mod(n\,10))" -vframes {num_frames} dataset_{name}/test/smile/test_smile_%04d.jpg`


The following MATLAB packages are required:
- Computer Vision Toolbox
- Image Processing Toolbox
- MATLAB Support Package for USB Webcams
- Data Acquisition Toolbox

The following python packages are required:
`python -m pip install --user torch torchvision pillow numpy`

Then, in `realtime_smile_detector.m`, update `pyenv` to the python path and update the `.pth` file to the correct model.

Finally, call `run_detector` in the MATLAB command window.
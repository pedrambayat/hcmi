from create_train_data import CreateTrainData
ffmpeg = r"ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
ffprobe = r"ffmpeg-8.0.1-full_build\bin\ffprobe.exe"
smile_path = r"C:\Users\fwlin\hcmi\dataset\dataset_pedram\smile.mp4"
frown_path = r"C:\Users\fwlin\hcmi\dataset\dataset_pedram\frown.mp4"

smiles = CreateTrainData(
    video_path=smile_path,
    label="smile",
    ffmpeg_path=ffmpeg,
    ffprobe_path=ffprobe
)

frowns = CreateTrainData(
    video_path=frown_path,
    label="frown",
    ffmpeg_path=ffmpeg,
    ffprobe_path=ffprobe
)

smiles.run()
frowns.run()
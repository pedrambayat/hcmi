from create_train_data import CreateTrainData
ffmpeg = "ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
ffprobe = "ffmpeg-8.0.1-full_build/bin/ffprobe.exe"
name = "hannah"
smile_path = f"dataset/dataset_{name}/smile.mp4"
frown_path = f"dataset/dataset_{name}/frown.mp4"
smiles = CreateTrainData(
    video_path=smile_path,
    label="smile",
    dataset_name=f"dataset_{name}",
    ffmpeg_path=ffmpeg,
    ffprobe_path=ffprobe
)

frowns = CreateTrainData(
    video_path=frown_path,
    label="frown",
    dataset_name=f"dataset_{name}",
    ffmpeg_path=ffmpeg,
    ffprobe_path=ffprobe
)

smiles.run()
frowns.run()
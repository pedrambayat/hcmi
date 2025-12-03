import subprocess
import os
import math

class CreateTrainData:
    def __init__(self, video_path, train_frames=300, test_frames=50,
                 train_folder="train", test_folder="test"):
        """
        video_path: full path to input video
        train_frames: number of frames to extract for training
        test_frames: number of frames to extract for testing
        train_folder: folder to save training frames
        test_folder: folder to save testing frames
        """
        self.video_path = video_path
        self.train_frames = train_frames
        self.test_frames = test_frames
        self.train_folder = train_folder
        self.test_folder = test_folder

        # create folders if they don't exist
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)

        # get total frames in video
        self.total_frames = self.get_total_frames()
        print(f"Total frames in video: {self.total_frames}")

    def get_total_frames(self):
        # get total number of frames with ffprobe
        cmd = [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            self.video_path
        ]
        total = int(subprocess.check_output(cmd).strip())
        return total

    def extract_frames(self, output_folder, num_frames):
        interval = max(1, math.floor(self.total_frames / num_frames))
        cmd = [
            "ffmpeg", "-i", self.video_path,
            "-vf", f"select=not(mod(n\\,{interval}))",
            "-vsync", "vfr",
            os.path.join(output_folder, "%04d.jpg")
        ]
        print(f"Extracting {num_frames} frames to '{output_folder}' every {interval} frames...")
        subprocess.run(cmd)

    def run(self):
        """Extracts both train and test frames"""
        self.extract_frames(self.train_folder, self.train_frames)
        self.extract_frames(self.test_folder, self.test_frames)
        print("Done extracting train and test frames!")
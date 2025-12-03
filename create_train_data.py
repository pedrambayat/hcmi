import subprocess
import os
import math

class CreateTrainData:
    def __init__(self, video_path, label, train_frames=300, test_frames=50,
                 train_folder="train", test_folder="test"):
        """
        video_path: full path to input video
        label: subfolder name ('smile' or 'frown')
        train_frames: number of frames to extract for training
        test_frames: number of frames to extract for testing
        train_folder: folder to save training frames
        test_folder: folder to save testing frames
        """
        self.video_path = video_path
        self.label = label
        self.train_frames = train_frames
        self.test_frames = test_frames
        self.train_folder = os.path.join(train_folder, label)
        self.test_folder = os.path.join(test_folder, label)

        # Create folders if they don't exist
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)

        # Get total frames in video
        self.total_frames = self.get_total_frames()
        print(f"Total frames in video: {self.total_frames}")

    def get_total_frames(self):
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
        self.extract_frames(self.train_folder, self.train_frames)
        self.extract_frames(self.test_folder, self.test_frames)
        print(f"Done extracting train and test frames for label '{self.label}'!")
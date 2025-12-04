import os
import subprocess

class CreateTrainData:
    def __init__(self, video_path, label, ffmpeg_path=None, ffprobe_path=None,
                 num_train=300, num_test=50):

        self.video_path = video_path
        self.label = label
        self.num_train = num_train
        self.num_test = num_test

        # Allow override of tool paths
        self.ffmpeg = ffmpeg_path or "ffmpeg"
        self.ffprobe = ffprobe_path or "ffprobe"

        # Output directories
        self.train_dir = os.path.join("dataset", "train", label)
        self.test_dir = os.path.join("dataset", "test", label)

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        self.total_frames = self.get_total_frames()

    def get_total_frames(self):
        """Use ffprobe to count frames."""
        cmd = [
            self.ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            self.video_path
        ]

        try:
            output = subprocess.check_output(cmd).decode().strip()
            return int(output)
        except FileNotFoundError:
            raise FileNotFoundError("ffprobe not found. Provide full path using ffprobe_path='C:/.../ffprobe.exe'")
        except Exception as e:
            raise RuntimeError(f"Failed to get frame count: {e}")

    def extract(self, num_frames, output_dir):
        """Extract a fixed number of evenly spaced frames."""
        # How often to grab a frame
        step = max(self.total_frames // num_frames, 1)

        cmd = [
            self.ffmpeg,
            "-i", self.video_path,
            "-vf", f"select='not(mod(n\\,{step}))'",
            "-vframes", str(num_frames),
            os.path.join(output_dir, f"{self.label}_%04d.jpg")
        ]

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found. Provide full path using ffmpeg_path='C:/.../ffmpeg.exe'")
        except Exception as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e}")

    def run(self):
        print(f"[INFO] Total frames in {self.label}: {self.total_frames}")
        print(f"[INFO] Extracting {self.num_train} → train/{self.label}/")
        self.extract(self.num_train, self.train_dir)

        print(f"[INFO] Extracting {self.num_test} → test/{self.label}/")
        self.extract(self.num_test, self.test_dir)

        print("[DONE] Frame extraction complete.")
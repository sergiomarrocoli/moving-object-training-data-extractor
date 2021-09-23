from dependencies import *

class VideoCombiner:
    def __init__(self, file_directory, subject_ids, percent_frames):
        self.file_directory = file_directory
        self.subject_ids = subject_ids
        self.percent_frames = percent_frames
    
    def calculate_frames(self, video, percent_frames):
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        no_frames = round(frame_count * (percent_frames / 100))
        frames = list(range(1, no_frames + 1))
        frames = [round(i * percent_frames) - 1 for i in frames]
        return frames
    
    def get_frames(self):
        images = []
        if self.is_colour():
            for video in self.subject_ids:
                if(os.path.isfile(f"{self.file_directory}/{video}.mp4")):
                    video = cv2.VideoCapture(f"{self.file_directory}/{video}.mp4")
                    frames = self.calculate_frames(video, self.percent_frames)
                    for frame in frames:
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        _, image = video.read()
                        images.append(image)
        return images
    
    def is_colour(self):
        print(f"getting file: {self.file_directory}/{self.subject_ids[0]}.mp4")
        if(os.path.isfile(f"{self.file_directory}/{self.subject_ids[0]}.mp4")):
            try:
                video = cv2.VideoCapture(f"{self.file_directory}/{self.subject_ids[0]}.mp4")
                video.set(cv2.CAP_PROP_POS_FRAMES, 10)
                _, image = video.read()
                hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                low_green = np.array([25, 52, 72])
                high_green = np.array([102, 255, 255])
                green_mask = cv2.inRange(hsv_frame, low_green, high_green)
                print(f"Colour: {sum(sum(green_mask)) > 100}")
                return sum(sum(green_mask)) > 1000
            except:
                return False
        else:
            return False

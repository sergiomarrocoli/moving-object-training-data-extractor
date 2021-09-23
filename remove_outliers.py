from dependencies import *

class RemoveOutliers:
    def __init__(self, frames, tolerance):
        self.frames = frames
        self.tolerance = tolerance
        
    def get_brightness(self):
        brightness = []
        for frame in self.frames:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            brightness.append(int(hsv_frame[:,:,1].mean()))
        return brightness
            
    def select_useable_frames(self, useable):
        keep = []
        for i in range(len(self.frames)):
            if useable[i]:
                keep.append(self.frames[i])
        return keep
        
    def get_frames(self):
        brightness = np.array(self.get_brightness())
        mode = stats.mode(brightness)[0][0]
        sd = np.std(brightness)
        min = mode - (sd * self.tolerance)
        max = mode + (sd * self.tolerance)
        over = min < brightness
        under = max > brightness
        useable = np.logical_and(over, under)
        good_frames = self.select_useable_frames(useable)
        return good_frames

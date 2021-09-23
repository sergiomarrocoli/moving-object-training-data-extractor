from dependencies import *

class ImageExporter:
    def __init__(self, frames, bounding_boxes, export_path, box_counter, frame_counter):
        self.frames = frames
        self.frame_counter = frame_counter
        self.bounding_boxes = bounding_boxes
        self.export_path = export_path
        self.box_counter = box_counter
        
    def export_images(self):
        print(f"print { len(sum(self.bounding_boxes, []))} images")
        box_ids = []
        for i in range(len(self.frames)):
            if len(self.bounding_boxes[i]) > 0:
                cv2.imwrite(f"{self.export_path}/frames/{self.frame_counter}.png", self.frames[i])
                for box in self.bounding_boxes[i]:
                    clipped_image = self.clip_image_by_bounding_box(self.frames[i], box)
                    self.save_image(self.box_counter, 'boxes', clipped_image)
                    box_ids.append([self.box_counter, self.frame_counter, box])
                    self.box_counter += 1
                self.frame_counter += 1
        return self.box_counter, self.frame_counter, box_ids
        
    def clip_image_by_bounding_box(self, image, box):
        clip = image[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
        return clip
        
    def save_image(self, id, folder, image):
        cv2.imwrite(f"{self.export_path}/{folder}/{id}.png", image)
        print(f"image {id} saved to {self.export_path}/{folder}/{id}.png")
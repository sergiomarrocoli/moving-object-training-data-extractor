from dependencies import *

class ImageProcessor:
	def __init__(self, frames, k_size):
		self.frames = frames
		self.k_size = k_size
		
	def analyse_frames(self):
		if len(self.frames) == 0:
			return [], [], []
		else:
			boxes = []
			converted_frames = self.convert_frames(self.frames)
			average = self.get_average(converted_frames)
			image_diffs = self.get_diffs(converted_frames, average)
			diff_masks = self.diffs_to_masks(image_diffs, self.k_size)
			boxes, _ = self.get_bounding_boxes(diff_masks)
			return boxes, image_diffs, diff_masks
		
	def get_frames(self, video, no_frames):
		frames = self.calculate_frames(video, no_frames)
		images = []
		for frame in frames:
			video.set(cv2.CAP_PROP_POS_FRAMES, frame)
			_, image = video.read()
			gauss_blur = cv2.GaussianBlur(image, (7, 7), 0)
			images.append(gauss_blur)
		print(f"Extracted {len(frames)} frames")
		return images
		
	def convert_frames(self, frames):
		float_frames = []
		for frame in frames:
			gauss_blur = cv2.GaussianBlur(frame, (self.k_size, self.k_size), 0)
			float_frame = gauss_blur.astype(np.float32) / 255
			float_frames.append(float_frame)
		return float_frames
		
	def get_average(self, frames):
		frames_shape = frames[0].shape
		height = frames_shape[0]
		width = frames_shape[1]
		if len(frames_shape) < 3:
			average_image = np.full((height, width), 0, np.float32)
		else:
			average_image = np.full((height, width, 3), 0, np.float32)
		no_frames = len(frames)
		for frame in frames:
			average_image += frame / no_frames
		print(f"Calculated mean: {average_image.shape}")
		return average_image
		
	def get_diffs(self, frames, average_image):
		colour = True if len(average_image.shape) > 1 else False
		differences = []
		if colour:
			for frame in frames:
				_, image_diff = compare_ssim(frame, average_image, multichannel=True, full=True)
				abs_diff = cv2.convertScaleAbs(image_diff)
				differences.append(image_diff)
		else:
			for frame in frames:
				image_diff = frame - average_image
				differences.append(abs_diff)
		print(f"Calculated differences: {len(differences)} frames, {differences[0].shape}")
		return differences
		
	def diffs_to_masks(self, frames, k_size):
		colour = len(frames[0].shape) > 2
		masks = []
		kernel = cv2.getGaussianKernel(k_size, 0)
		mask_type = ""
		if colour:
			mask_type = "colour"
			for frame in frames:
				bw_frame = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
				_, binary = cv2.threshold(bw_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				mask = cv2.erode(binary, kernel, iterations = 1)
				masks.append(mask.astype('uint8'))
		print(f"Calculated {len(masks)} {mask_type} masks, {masks[0].shape}, {len(masks)} frames")
		return masks
		
	def get_bounding_boxes(self, frames):
		boxes = []
		max_in_one_frame = 0
		for frame in frames:
			frame_boxes = self.get_bounding_boxes_for_single_frame(frame)
			boxes.append(frame_boxes)
		return boxes, max_in_one_frame
		
	def get_bounding_boxes_for_single_frame(self, frame):
		_, contours, hierarchy = cv2.findContours(~frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		bounding_boxes = []
		for contour in contours:
			contour_area = cv2.contourArea(contour)
			if 1750 < contour_area < 450000:
				x,y,w,h = cv2.boundingRect(contour)
				bounding_boxes.append([x, y, w, h])
		if len(bounding_boxes) > 1:
			bounding_boxes = self.group_boxes(bounding_boxes)
		return bounding_boxes
		
	def group_boxes(self, bounding_boxes):
		boxes = bounding_boxes
		again = True
		while again == True:
			again = False
			if len(boxes) > 1:
				for i in range(len(boxes)):
					for j in range(len(boxes)):
						if (i != j):
							if boxes[i] != False and boxes[j] != False:
								if self.boxes_overlap(boxes[i], boxes[j]):
									boxes[i] = self.combine_boxes(boxes[i], boxes[j])
									boxes[j] = False
									again = True
				boxes = list(filter(lambda x: x != False, boxes))
		return boxes
		
	def boxes_overlap(self, box_1, box_2):
		if self.horizontal_overlap(box_1, box_2) and self.vertical_overlap(box_1, box_2):
			return True
		else:
			return False
			
	def horizontal_overlap(self, box_1, box_2):
		box_1_left = box_1[0]
		box_1_right = box_1[0] + box_1[2]
		box_2_left = box_2[0]
		box_2_right = box_2[0] + box_1[2]
		return (box_2_left <= box_1_left <= box_2_right) or (box_2_left <= box_1_right <= box_2_right)
	
	def vertical_overlap(self, box_1, box_2):
		box_1_left = box_1[1]
		box_1_right = box_1[1] + box_1[3]
		box_2_left = box_2[1]
		box_2_right = box_2[1] + box_1[3]
		return (box_2_left <= box_1_left <= box_2_right) or (box_2_left <= box_1_right <= box_2_right)
		
	def combine_boxes(self, box_1, box_2):
		x = min(box_1[0], box_2[0])
		max_x = max((box_1[0] + box_1[2]), (box_2[0] + box_2[2]))
		y = min(box_1[1], box_2[1])
		max_y = max((box_1[1] + box_1[3]), (box_2[1] + box_2[3]))
		w = max_x - x
		h = max_y - y
		return [x, y, w, h]
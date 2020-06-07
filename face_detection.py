import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

class FaceDetection():
    def __init__(self):
        # Default parameters. DO NOT CHANGE
        self.model_def = './yolov3/config/yolov3-face.cfg'
        self.weights_path = './yolov3/weights/yolov3_face.weights'
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.img_size = 416

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up model
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)

        # Load weights
        self.model.load_darknet_weights(self.weights_path)

        # Set in evaluation mode
        self.model.eval()

        # CPU or Cuda
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def detect(self, image):
        tensor_image = self.transform_image(image)

        # Get detections
        with torch.no_grad():
            detections = self.model(tensor_image)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]

        if detections is None:
            return None
        else:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, self.img_size, image.shape[:2])

            # Return the biggest box
            biggest_box = None
            biggest_box_area = 0

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area > biggest_box_area:
                    biggest_box_area = area
                    biggest_box = {
                        'x1': x1.item(),
                        'y1': y1.item(),
                        'x2': x2.item(),
                        'y2': y2.item(),
                        'width': width.item(),
                        'height': height.item()
                    }

            return biggest_box
        
    def plot(self, image, bbox):
        if bbox is not None:
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            plot_bbox = patches.Rectangle((bbox['x1'], bbox['y1']), bbox['width'], bbox['height'], linewidth=2, edgecolor='white', facecolor="none")
            ax.add_patch(plot_bbox)
            plt.show()

    def transform_image(self, image):
        # To tensor
        image = transforms.ToTensor()(image)

        # Pad the image to have square dimensions
        image, _ = pad_to_square(image, 0)

        # Resize to YOLOv3 CNN
        image = resize(image, self.img_size).unsqueeze(0)

        # Convert to CUDA or CPU
        image = Variable(image.type(self.Tensor))

        return image
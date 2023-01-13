import os
import cv2
import warnings
import torch
import torchvision
import matplotlib.image as im

from pathlib import Path

warnings.filterwarnings('ignore')


class CheckboxDetectionTest:
    # The init method or constructor
    def __init__(self, input_file, output_dir, model_path):
        # Instance Variable
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_path = model_path

    """  checkbox detection """

    def checkbox_detection(self):
        detection_path = "checkbox_detection"
        detection_path = os.path.join(self.output_dir, detection_path)
        os.makedirs(detection_path, exist_ok=True)

        # get a file name and file extension
        file_name = Path(self.input_file).stem
        file_extension = Path(self.input_file).suffix

        detection_path = os.path.join(detection_path, file_name)
        os.makedirs(detection_path, exist_ok=True)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(self.model_path)
        # model.load_state_dict(torch.load(self.model_path))
        model.eval()

        input_image = im.imread(self.input_file)
        input_img = torchvision.transforms.functional.to_tensor(input_image).to(device)

        checkbox_tensors = model([input_img])[0]["boxes"]
        prediction = checkbox_tensors.data.cpu().numpy()

        count = 0
        for table_tensor in prediction:
            table_tensor = [int(i) for i in table_tensor]
            detected_file = f'{detection_path}/{file_name}_{count}.jpg'
            cv2.imwrite(detected_file,input_image[table_tensor[1]:table_tensor[3], table_tensor[0]:table_tensor[2]])
            count += 1

        return detected_file



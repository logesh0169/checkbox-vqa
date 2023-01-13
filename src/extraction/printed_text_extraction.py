import os
import warnings

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils.logger import logger

warnings.filterwarnings('ignore')


class PrintedTextExtraction:

    # The init method or constructor
    def __init__(self, input_file):

        # Instance Variable
        self.input_file = input_file

    """   Printed Text Extraction using TrOCR """

    def printed_text_extraction(self):

        logger.info("PRINTED TEXT EXTRACTION START")

        image = Image.open(self.input_file).convert("RGB")

        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        extracted_printed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info("PRINTED TEXT EXTRACTION END")

        return extracted_printed_text




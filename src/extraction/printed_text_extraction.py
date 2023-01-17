import warnings

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils.logger import logger
from transformers import pipeline

warnings.filterwarnings('ignore')


class PrintedTextExtraction:

    # The init method or constructor
    def __init__(self, input_file, model):

        # Instance Variable
        self.input_file = input_file
        self.model = model

    """   Printed Text Extraction using TrOCR """

    def printed_text_extraction(self):

        logger.info("PRINTED TEXT EXTRACTION START")

        image = Image.open(self.input_file).convert("RGB")

        processor = TrOCRProcessor.from_pretrained(self.model)
        model = VisionEncoderDecoderModel.from_pretrained(self.model)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        extracted_printed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")
        #
        # extracted_printed_text = fix_spelling(extracted_printed_text, max_length=2048)
        extracted_printed_text = ''.join((x for x in extracted_printed_text if not x.isdigit()))
        logger.info("PRINTED TEXT EXTRACTION END")

        return extracted_printed_text




import warnings

from pathlib import Path
from utils.config import *
from utils.file_info import *
from utils.logger import logger

from src.detection.checkbox_detection_test import CheckboxDetectionTest
from src.recognition.checkbox_state_recognition_test import CheckboxStateRecognition
from src.extraction.printed_text_extraction import PrintedTextExtraction

warnings.filterwarnings('ignore')


# main program
if __name__ == '__main__':

    logger.info("CHECKBOX VQA STARTED")

    res = {}

    file_name = Path(INPUT_FILE).stem
    file_extension = Path(INPUT_FILE).suffix
    file_size = get_file_size(INPUT_FILE)
    file_checksum = get_checksum(INPUT_FILE)

    checkbox_detection = CheckboxDetectionTest(INPUT_FILE, OUTPUT_DIR, CD_MODEL_PATH)
    detected_file = checkbox_detection.checkbox_detection()

    checkbox_state_recognition = CheckboxStateRecognition(detected_file, CR_MODEL_PATH, CR_IMG_WIDTH, CR_IMG_HEIGHT)
    checkbox_state = checkbox_state_recognition.checkbox_state_recognition()

    printed_text_extraction = PrintedTextExtraction(INPUT_FILE)
    extracted_printed_text = printed_text_extraction.printed_text_extraction()

    res['inputFilePath'] = INPUT_FILE
    res['inputFileName'] = file_name
    res['fileExtension'] = file_extension
    res['inputFileSize'] = file_size
    res['inputFileChecksum'] = file_checksum
    res['checkboxDetectedFile'] = detected_file
    res['checkbox_state'] = checkbox_state
    res['extracted_printed_text'] = extracted_printed_text

    print(res)

    logger.info("CHECKBOX VQA END")







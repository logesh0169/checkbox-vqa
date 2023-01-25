import warnings

from pathlib import Path
from utils.config import *
from utils.file_info import *
from utils.logger import logger

from src.detection.checkbox_detection_test import CheckboxDetectionTest
from src.detection.checkbox_attr_detection_test import CheckboxAttrDetectionTest
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

    checkbox_attr_detection = CheckboxAttrDetectionTest(INPUT_FILE, SOURCE_OF_TRUTH_DIR, CAD_MODEL_PATH)
    checkbox_attr_detected_path = checkbox_attr_detection.checkbox_attr_detection()

    extracted_text = []

    for checkbox_attr_detected_file in os.listdir(checkbox_attr_detected_path):

        checkbox_detection = CheckboxDetectionTest(checkbox_attr_detected_path+'/'+checkbox_attr_detected_file, SOURCE_OF_TRUTH_DIR, CD_MODEL_PATH)
        detected_path = checkbox_detection.checkbox_detection()

        printed_text_extraction = PrintedTextExtraction(checkbox_attr_detected_path+'/'+checkbox_attr_detected_file, PRINTED_TEXT_DETECTION_SMALL_MODEL)
        extracted_printed_text = printed_text_extraction.printed_text_extraction()
        extracted_text.append({checkbox_attr_detected_file: extracted_printed_text})

    state_reg = []

    for detected_file in os.listdir(detected_path):
        checkbox_state_recognition = CheckboxStateRecognition(detected_path+'/'+detected_file, CR_MODEL_PATH, CR_IMG_WIDTH, CR_IMG_HEIGHT)
        checkbox_state = checkbox_state_recognition.checkbox_state_recognition()

        state_reg.append(checkbox_state)

    res['inputFilePath'] = INPUT_FILE
    res['inputFileName'] = file_name
    res['fileExtension'] = file_extension
    res['inputFileSize'] = file_size
    res['inputFileChecksum'] = file_checksum
    res['checkbox_state'] = state_reg
    res['extracted_printed_text'] = extracted_text

    print(res)
    logger.info("CHECKBOX VQA END")

import os

# parent directory
PARENT_DIR = "/home/hwuser/Workspace/"

# output directory
OUTPUT_DIR = "dev/prod-dev/checkbox-vqa/v1.0/output/"
OUTPUT_DIR = os.path.join(PARENT_DIR, OUTPUT_DIR)

# source of truth
SOURCE_OF_TRUTH_DIR = "source_of_truth/"
SOURCE_OF_TRUTH_DIR = os.path.join(OUTPUT_DIR, SOURCE_OF_TRUTH_DIR)

# system of records
SYSTEM_OF_RECORDS_DIR = "system_of_records/"
SYSTEM_OF_RECORDS_DIR = os.path.join(OUTPUT_DIR, SYSTEM_OF_RECORDS_DIR)

# input file directory

SOURCE_OF_ORIGIN = "/home/hwuser/Workspace/dev/prod-dev/checkbox-vqa/v1.0/source_of_origin/test.jpg"

INPUT_FILE = os.path.join(PARENT_DIR, SOURCE_OF_ORIGIN)


# checkbox detection hyper_parameters

CD_IMG_WIDTH = 50
CD_IMG_HEIGHT = 50

CD_MODEL_PATH = os.path.join(OUTPUT_DIR, "model/checkbox_state_recognition_model_v1.0.h5")

# checkbox recognition hyper_parameters

CR_IMG_WIDTH = 50
CR_IMG_HEIGHT = 50

CR_MODEL_PATH = os.path.join(OUTPUT_DIR, "model/checkbox_state_recognition_model_v1.0.h5")

##download_backblaze_data.py###################################################
#
# Requirements
#
#    -  wget (`pip install wget`)
#
# How to use:
#
# `python download_backblaze_data.py [parameter]`
#
# Parameters:
#
#    - parameter: either `full` or `min`
#
#    If set to `full` will download the data sets used in Chapter 7 (4 files,
#    ~2.3GB compressed, 12.4GB uncompressed).
#
#    If set to `minimal` will download only 2019 Q3 (1 file, 574MB compressed,
#    3.1GB uncompressed).
#
###############################################################################

import sys
import wget


DATASETS_FULL = [
    "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2019.zip",
    "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2019.zip",
    "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q3_2019.zip",
    "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2019.zip",
]

DATASETS_MINIMAL = DATASETS_FULL[2:3]  # Slice to keep as a list. Simplifies
# the code later.

if __name__ == "__main__":

    try:
        param = sys.argv[1]

        if param.lower() == "full":
            datasets = DATASETS_FULL
        elif param.lower() == "minimal":
            datasets = DATASETS_MINIMAL
        else:
            raise AssertionError()
    except (AssertionError, IndexError):
        print(
            "Parameter missing. Refer to the documentation at the top of the source code for more information"
        )
        sys.exit(1)

    for dataset in datasets:
        print("\n", dataset.split("/")[-1])
        wget.download(dataset, out="../../data/Ch07/")

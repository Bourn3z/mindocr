# The data preprocess customized for inference needs to be imported here

from deploy.py_infer.src.data_process.preprocess.transforms import *  # noqa

gear_supported_list = [
    "DetResize",
    "DetResizeNormForInfer",
    "SVTRRecResizeImg",
    "RecResizeNormForInfer",
    "RecResizeNormForViTSTR",
    "RecResizeNormForMMOCR",
]

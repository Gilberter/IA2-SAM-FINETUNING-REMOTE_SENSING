

import os
from dotenv import load_dotenv
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


load_dotenv()
access_token = os.getenv("access_token")

model = build_sam3_image_model()
processor = Sam3Processor(model)

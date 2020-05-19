import os
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import json
import cv2
import transforms
import matplotlib.pyplot as plt
from dataset import *









if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
        
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    

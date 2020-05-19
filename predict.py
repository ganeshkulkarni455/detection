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
import utils
from dataset import *

def get_prediction(img_path, threshold):
  
  img = Image.open(img_path) # Load the image
  #transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  #img, _ = transform(img) # Apply the transform to the image
  trans1 = transforms.ToTensor()
  img, _ = trans1(img,_)
  pred = model([img.to(device)]) # Pass the image to the model
  pred_class = [NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class

def object_detection_api(img_path, path,threshold=0.5, rect_th=3, text_size=3, text_th=3):
  
  boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
  img = cv2.imread(img_path) # Read image with cv2
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    
  cv2.imwrite(path, img)
#   plt.figure(figsize=(20,30)) # display the output image
#   plt.imshow(img)
#   plt.xticks([])
#   plt.yticks([])
#   plt.show()









if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
        
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    
    
    
    
    args = parser.parse_args()
    
    if args.output_dir:
     utils.mkdir(args.output_dir)
    
    main(args)

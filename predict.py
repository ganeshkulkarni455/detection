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
from torchvision import transforms
import matplotlib.pyplot as plt
import utils
from datasets import *

def get_prediction(model, device, img_path, threshold):
  
  img = Image.open(img_path) # Load the image
  #transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  #img, _ = transform(img) # Apply the transform to the image
  img_transforms = transforms.Compose([transforms.ToTensor()])
  img = img_transforms(img)
  pred = model([img.to(device)]) # Pass the image to the model
  pred_class = [NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class

def object_detection_api(model, device, img_path, output_image_path, threshold, rect_th, text_size, text_th):
  
  boxes, pred_cls = get_prediction(model, device, img_path, threshold) # Get predictions
  img = cv2.imread(img_path) # Read image with cv2
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    
  cv2.imwrite(output_image_path, img)
#   plt.figure(figsize=(20,30)) # display the output image
#   plt.imshow(img)
#   plt.xticks([])
#   plt.yticks([])
#   plt.show()

def main(args):
  
  print(args)
  device = torch.device(args.device)
  
  NAMES, num_classes = get_names(args.json_file_path)
  
  print("Loading model")

  model = torchvision.models.detection.__dict__[args.model](pretrained=False)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  model.load_state_dict(torch.load(args.weight_file_path, map_location=args.device)) # Initialisng Model with loaded weights
  
  model.to(device)
  
  model.eval()
  
  imgs = list(sorted(os.listdir(args.image_folder_path)))
  for i in range (0, len(imgs)):
    img_path = os.path.join(args.image_folder_path, imgs[i])
    output_image_path = os.path.join(args.output_dir, imgs[i])
    object_detection_api(model, args.device, img_path, output_image_path, args.threshold, args.rect_th, args.text_size, args.text_th)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
   
    parser.add_argument('--image_folder_path', type= str, help = 'path directory of image folder for prediction')
    parser.add_argument('--json_file_path', type= str, help = 'path directory of json file of coco dataset format')
    parser.add_argument('--weight_file_path', type= str, help = 'path directory of weight file of .pth format')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--threshold', default=0.5, type=float, help='threshold value')
    parser.add_argument('--rect_th', default=3, type=int, help='rect_th value')
    parser.add_argument('--text_size', default=3, type=int, help='text_size value')
    parser.add_argument('--text_th', default=3, type=int, help='text_th value')
    
    args = parser.parse_args()
    
    if args.output_dir:
     utils.mkdir(args.output_dir)
    
    main(args)

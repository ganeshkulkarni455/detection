import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json


class CustomeCocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path, json_file_path, transforms=None):
        #self.root = root
        self.image_folder_path = image_folder_path
        self.json_file_path = json_file_path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(image_folder_path)))
        self.anno = json.load(open(json_file_path))
        self.annotation = self.anno["annotations"]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        target = {}
        boxes = []
        labels = []
        area = []
        image_id = []
        iscrowd = []

        for j in range (0,len(self.annotation)):
          if self.annotation[j]["image_id"] == idx :

            self.annotation[j]["bbox"][2] = self.annotation[j]["bbox"][0] + self.annotation[j]["bbox"][2]
            self.annotation[j]["bbox"][3] = self.annotation[j]["bbox"][1] + self.annotation[j]["bbox"][3]

            boxes.append(self.annotation[j]["bbox"])
            labels.append(self.annotation[j]["category_id"]+1)
            area.append(self.annotation[j]["area"])
            iscrowd.append(self.annotation[j]["iscrowd"])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd 

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
        
def get_names(json_file_path):

  anno = json.load(open(json_file_path))
  categories = anno["categories"]
  NAME = ["__background__"]
  for i in range (0,len(categories)):
    NAME.append(str(categories[i]["name"]))

  # num_classes include background
  num_classes = len(NAME)

  return NAME, num_classes 

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path, json_file_path, transforms=None):
        #self.root = root
        self.image_folder_path = image_folder_path
        self.mask_folder_path = json_file_path              #give mask folder directory as json for perticular this dataset
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(image_folder_path)))
        self.masks = list(sorted(os.listdir(json_file_path)))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.image_folder_path, self.imgs[idx])
        mask_path = os.path.join(self.mask_folder_path", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

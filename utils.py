from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os

import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    
    
# class LoadImagesAndLabels:  # for training/testing
#     def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
#                  cache_labels=True, cache_images=False, single_cls=False):
#         path = str(Path(path))  # os-agnostic
#         assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
#         with open(path, 'r') as f:
#             self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
#                               if os.path.splitext(x)[-1].lower() in img_formats]

#         n = len(self.img_files)
#         assert n > 0, 'No images found in %s. See %s' % (path, help_url)
#         bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
#         nb = bi[-1] + 1  # number of batches

#         self.n = n
#         self.batch = bi  # batch index of image
#         self.img_size = img_size
#         self.augment = augment
#         self.hyp = hyp
#         self.image_weights = image_weights
#         self.rect = False if image_weights else rect
#         self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

#         # Define labels
#         self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
#                             for x in self.img_files]

#         # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
#         if self.rect:
#             # Read image shapes (wh)
#             sp = path.replace('.txt', '.shapes')  # shapefile path
#             try:
#                 with open(sp, 'r') as f:  # read existing shapefile
#                     s = [x.split() for x in f.read().splitlines()]
#                     assert len(s) == n, 'Shapefile out of sync'
#             except:
#                 s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
#                 np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)

#             # Sort by aspect ratio
#             s = np.array(s, dtype=np.float64)
#             ar = s[:, 1] / s[:, 0]  # aspect ratio
#             i = ar.argsort()
#             self.img_files = [self.img_files[i] for i in i]
#             self.label_files = [self.label_files[i] for i in i]
#             self.shapes = s[i]  # wh
#             ar = ar[i]

#             # Set training image shapes
#             shapes = [[1, 1]] * nb
#             for i in range(nb):
#                 ari = ar[bi == i]
#                 mini, maxi = ari.min(), ari.max()
#                 if maxi < 1:
#                     shapes[i] = [maxi, 1]
#                 elif mini > 1:
#                     shapes[i] = [1, 1 / mini]

#             self.batch_shapes = np.ceil(np.array(shapes) * img_size / 64.).astype(np.int) * 64

#         # Preload labels (required for weighted CE training)
#         self.imgs = [None] * n
#         self.labels = [None] * n
#         if cache_labels or image_weights:  # cache labels for faster training
#             self.labels = [np.zeros((0, 5))] * n
#             extract_bounding_boxes = False
#             create_datasubset = False
#             pbar = tqdm(self.label_files, desc='Caching labels')
#             nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
#             for i, file in enumerate(pbar):
#                 try:
#                     with open(file, 'r') as f:
#                         l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
#                 except:
#                     nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
#                     continue

#                 if l.shape[0]:
#                     assert l.shape[1] == 5, '> 5 label columns: %s' % file
#                     assert (l >= 0).all(), 'negative labels: %s' % file
#                     assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
#                     if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
#                         nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
#                     if single_cls:
#                         l[:, 0] = 0  # force dataset into single-class mode
#                     self.labels[i] = l
#                     nf += 1  # file found

#                     # Create subdataset (a smaller dataset)
#                     if create_datasubset and ns < 1E4:
#                         if ns == 0:
#                             create_folder(path='./datasubset')
#                             os.makedirs('./datasubset/images')
#                         exclude_classes = 43
#                         if exclude_classes not in l[:, 0]:
#                             ns += 1
#                             # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
#                             with open('./datasubset/images.txt', 'a') as f:
#                                 f.write(self.img_files[i] + '\n')

#                     # Extract object detection boxes for a second stage classifier
#                     if extract_bounding_boxes:
#                         p = Path(self.img_files[i])
#                         img = cv2.imread(str(p))
#                         h, w = img.shape[:2]
#                         for j, x in enumerate(l):
#                             f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
#                             if not os.path.exists(Path(f).parent):
#                                 os.makedirs(Path(f).parent)  # make new output folder

#                             b = x[1:] * [w, h, w, h]  # box
#                             b[2:] = b[2:].max()  # rectangle to square
#                             b[2:] = b[2:] * 1.3 + 30  # pad
#                             b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

#                             b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
#                             b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
#                             assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
#                 else:
#                     ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
#                     # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

#                 pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
#                     nf, nm, ne, nd, n)
#             assert nf > 0, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)

#         # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
#         if cache_images:  # if training
#             gb = 0  # Gigabytes of cached images
#             pbar = tqdm(range(len(self.img_files)), desc='Caching images')
#             self.img_hw0, self.img_hw = [None] * n, [None] * n
#             for i in pbar:  # max 10k images
#                 self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
#                 gb += self.imgs[i].nbytes
#                 pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

#         # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
#         detect_corrupted_images = False
#         if detect_corrupted_images:
#             from skimage import io  # conda install -c conda-forge scikit-image
#             for file in tqdm(self.img_files, desc='Detecting corrupted images'):
#                 try:
#                     _ = io.imread(file)
#                 except:
#                     print('Corrupted image detected: %s' % file)

#     def __len__(self):
#         return len(self.img_files)

#     # def __iter__(self):
#     #     self.count = -1
#     #     print('ran dataset iter')
#     #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
#     #     return self

#     def __getitem__(self, index):
#         if self.image_weights:
#             index = self.indices[index]

#         hyp = self.hyp
#         if self.mosaic:
#             # Load mosaic
#             img, labels = load_mosaic(self, index)
#             shapes = None

#         else:
#             # Load image
#             img, (h0, w0), (h, w) = load_image(self, index)

#             # Letterbox
#             shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
#             img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
#             shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

#             # Load labels
#             labels = []
#             x = self.labels[index]
#             if x is not None and x.size > 0:
#                 # Normalized xywh to pixel xyxy format
#                 labels = x.copy()
#                 labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
#                 labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
#                 labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
#                 labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

#         if self.augment:
#             # Augment imagespace
#             if not self.mosaic:
#                 img, labels = random_affine(img, labels,
#                                             degrees=hyp['degrees'],
#                                             translate=hyp['translate'],
#                                             scale=hyp['scale'],
#                                             shear=hyp['shear'])

#             # Augment colorspace
#             augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

#             # Apply cutouts
#             # if random.random() < 0.9:
#             #     labels = cutout(img, labels)

#         nL = len(labels)  # number of labels
#         if nL:
#             # convert xyxy to xywh
#             labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

#             # Normalize coordinates 0 - 1
#             labels[:, [2, 4]] /= img.shape[0]  # height
#             labels[:, [1, 3]] /= img.shape[1]  # width

#         if self.augment:
#             # random left-right flip
#             lr_flip = True
#             if lr_flip and random.random() < 0.5:
#                 img = np.fliplr(img)
#                 if nL:
#                     labels[:, 1] = 1 - labels[:, 1]

#             # random up-down flip
#             ud_flip = False
#             if ud_flip and random.random() < 0.5:
#                 img = np.flipud(img)
#                 if nL:
#                     labels[:, 2] = 1 - labels[:, 2]

#         labels_out = torch.zeros((nL, 6))
#         if nL:
#             labels_out[:, 1:] = torch.from_numpy(labels)

#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)

#         return torch.from_numpy(img), labels_out, self.img_files[index], shapes
   

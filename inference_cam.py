import json
import math
import os
import time
from shutil import copyfile
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from datasets import get_loader
from datasets.dataset import pil_loader
from datasets.transform import get_transform

from models import resnet
from models.SSLTNet import SSLTNet
from models.clustering import compute_clusters
from models.logger import setup_logger
from models.loss import SSLTLoss
from models.option import get_args_parser
from models.util import AverageMeter
from models.lr_scheduler import get_scheduler

import matplotlib.pyplot as plt

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def build_clusters(encoder, dataloader, args):
    initial_clusters = compute_clusters(dataloader, encoder, args)
    return initial_clusters


def build_model(encoder, initial_clusters, args):
    model = SSLTNet(args, encoder, initial_clusters=initial_clusters).cuda()
    model = DistributedDataParallel(model, device_ids=[
                                    args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    return model


def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(
        f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def inference(model, args):
    model.eval()
    
    data_list = os.listdir(args.data_dir)
    iterator = tqdm(data_list, position=1)
    iterator.set_description('Generate Cluster Attention Map...')
    os.makedirs(f"{args.output_dir}/visualize_cam/", exist_ok=True)
    plt.figure()

    transforms = get_transform('val', args.crop) 
    im_loader = pil_loader
    for i, img_path in enumerate(iterator):
        img_path = os.path.join(args.data_dir, img_path)
        img = im_loader(img_path)
        input_image = transforms(img).unsqueeze(0)
        input_images = input_image.to(opt.device)
        out = model(input_images)
        CAM_batch = out['cam'].cpu().detach().numpy()

        q = args.num_queries
        nrows = math.ceil(math.sqrt(q + 1))
        ncols = int(math.sqrt(q + 1))

        if (nrows * ncols < q + 1): nrows += 1
        for j, cam in enumerate(CAM_batch):
            plt.clf()
            plt.subplot(nrows, ncols, 1)
            plt.axis('off')
            plt.imshow(img)
            for idx, am in enumerate(cam):
                plt.subplot(nrows, ncols, idx + 2)
                plt.axis('off')
                plt.imshow(am)
            plt.savefig(f"{args.output_dir}/visualize_cam/{i:05d}.jpg")
            
def main(args):

    # Build encoder
    encoder = resnet.__dict__[args.backbone](
        low_dim=args.feature_dim, head_type='early_return')
    encoder.to(args.device)
    # Build cluster
    initial_cluster = None

    # Build model
    model = build_model(encoder, initial_cluster, args)

    # optionally resume from a checkpoint
    assert os.path.isfile(args.pretrained_model)
    load_pretrained(model, args.pretrained_model)
    model.eval()
    inference(model, args)


if __name__ == '__main__':
    opt = get_args_parser().parse_args()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir,
                          distributed_rank=dist.get_rank(), name="sslt")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(opt)).items()))
    )

    main(opt)

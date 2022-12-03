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
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from datasets import get_loader

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


def inference(model, dataloader):
    model.eval()
    dataloader.dataset.random_sample = False

    iterator = tqdm(dataloader, position=1)
    iterator.set_description('Computing features & Clustering... ')
    plt.figure()
    for i, input_image in enumerate(iterator):
        input_images = input_image.to(opt.device)
        # [B, C, H, W]
        img_feat = model.backbone(input_images)
        out = model(input_image)
        # [B, Q, C]
        cluster_prototypes = out['cluster_prototypes'] 
        bs, q, _ = cluster_prototypes.size()
        # calculate cluster attention map
        CAM_batch = torch.einsum('bqc,bchw->bqhw', cluster_prototypes, img_feat)
        
        nrows = math.ceil(math.sqrt(q + 1)) 
        ncols = int(math.sqrt(q + 1))

        if (nrows * ncols < q + 1): nrows += 1
        plt.clf()
        plt.subplot(nrows, ncols, 1)
        for cam in CAM_batch:
            for idx, am in enumerate(cam):
                plt.subplot(nrows, ncols, idx + 2)
                plt.imshow(am)
        plt.savefig(f"{i:05d}.jpg")
            
def main(args):
    val_loader = get_loader(
        args.aug, args,
        prefix='val')
    args.num_instances = len(val_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

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


def train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        im_batch = data.cuda(non_blocking=True)

        loss = model(im_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # update meters and print info
        loss_meter.update(loss.item(), im_batch.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')

            # tensorboard logger
            if summary_writer is not None:
                step = (epoch - 1) * len(train_loader) + idx
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', loss_meter.val, step)


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

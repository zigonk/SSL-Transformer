import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import IMG_EXTENSIONS, DatasetFolder, default_img_loader


def get_loader(aug_type, args, prefix='train', batch_size=None):
    
    transform = get_transform(aug_type, args.crop, args.image_size)

    if (batch_size is None):
        batch_size = args.batch_size


    train_folder = os.path.join(args.data_dir, prefix)
    train_dataset = DatasetFolder(
        train_folder,
        default_img_loader,
        IMG_EXTENSIONS,
        transform=transform)

    # sampler
    sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True)
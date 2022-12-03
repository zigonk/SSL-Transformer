import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set self-supervised transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd',
                        help='for optimizer choice.')
    
    # Training parameters
    parser.add_argument('--pretrained-model', type=str, default="", help="pretrained model path")
    parser.add_argument('--auto_resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--verbose', action='store_true')
    

    # Model parameters
    parser.add_argument('--model', default='SSLTNet', type=str,
                        help="Name of model to use")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--feature_dim', type=int, default=256, help='feature dimension')
    parser.add_argument('--head_type', type=str, default='mlp_head', help='choose head type')
    parser.add_argument('--dec_type', type=str, choices=['trans-dec', 'cross-attn'], default='trans-dec',
                        help='for decoder type choice.')

    # * Transformer
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # Dataset parameters
    parser.add_argument('--data_dir', default='coco')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')

    # Clustering parameters
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pca_dim',      default=128,
                        type=int, help='Use sobel filter as initialization.')

    # Augmentation parameters
    parser.add_argument('--aug', type=str, default='NULL',
                    choices=['NULL', 'InstDisc', 'MoCov2', 'SimCLR', 'RandAug', 'BYOL', 'val'],
                    help='which augmentation to use.')
    parser.add_argument('--crop', type=float, default=0.08)

    # Weighted loss parameters
    parser.add_argument('--weight_ce', type=float, default=1)
    parser.add_argument('--weight_contrast', type=float, default=0)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    return parser

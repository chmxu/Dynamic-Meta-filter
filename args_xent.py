import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet_load')
    parser.add_argument('--root', type=str, default='/youtu_action_data/chengmingxu/data/miniImageNet_pickle')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--load', default=True)
    parser.add_argument('--cifar', default=False)
    parser.add_argument('--tiered', default=False)

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.35, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=90, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
                        help="stepsize to decay learning rate")

    parser.add_argument('--train-batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=4, type=int,
                        help="test batch size")

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--num_classes', type=int, default=64)
    parser.add_argument('--groups', type=int, default=64)
    parser.add_argument('--kernel', type=int, default=1)

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./weights')
    parser.add_argument('--gpu-devices', default='2', type=str)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')

    parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='val', type=str,
                        help='use test or val dataset to early stop')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', type=str, default='./weights/mini/1shot.pth.tar', metavar='PATH')

    return parser


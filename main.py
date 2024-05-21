import os
import time
import torch
import argparse
from torch.backends import cudnn

from models import build_net
from train import _train
from evaluate import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_path + '/'):
        os.makedirs('results/' + args.model_path +'/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus','dwt'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # Test
    parser.add_argument('--test_model', type=str, default='weights/MIMO-UNet.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    # dataloader
    parser.add_argument('--subset', type=str, default='Hday2night')
    parser.add_argument('--archive', type=str, default='datasets/ihm4/IHD_train_256.h5')
    parser.add_argument('--use_subarch', type=bool, default=True, choices=[True, False])
    parser.add_argument('--model_path', type=str, default='001')


    args = parser.parse_args()
    args.model_path = args.model_name + time.strftime('_%y%m%d%H%M%S', time.localtime())
    print(args.model_path)

    args.model_save_dir = os.path.join('results/', args.model_path, 'weights/')
    args.result_dir = os.path.join('results/', args.model_path, 'result_image/')
    print(args)
    main(args)

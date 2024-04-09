import argparse 
import torch 
import numpy as np
import os
from time import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument("--test_model_path", default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--decoder_type', default='E2E-DUNet', type=str, help='reconstruction network type')
    parser.add_argument('--cr', default=0.25, type=float, help='0.01, 0.04, 0.10, 0.25, 0.5')   
    parser.add_argument('--size', default=[256,256], help='gray: [256,256]; color: [321,481]')
    parser.add_argument('--meas_size', default=[128,128], help='gray: [26,26], [51,51], [81,81], [128,128], [181,181]; color: [32,48], [64,96], [102,152], [161,240], [227,340]')     
    parser.add_argument('--matrix_train', default=True, type=bool)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--color_channels', default=1,type=int)
    parser.add_argument('--stages', default=7,type=int)
    parser.add_argument('--channels', default=64,type=int)
    parser.add_argument('--mid_blocks', default=1)
    parser.add_argument('--enc_blocks', default=[1,1])
    parser.add_argument('--dec_blocks', default=[1,1])
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=200, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=100, type=int)
    parser.add_argument('--test_flag', default=True, type=bool)
    parser.add_argument('--train_data_path',type=str,default='/data/BSD400')
    parser.add_argument('--test_data_path',type=str,default='/data/Set11', help='gray: Set11; color: CBSD68')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile',nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

    args = parser.parse_args()

    return args

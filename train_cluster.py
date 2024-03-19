
from dpdl_defaults_cluster import *
import os

import torch
import numpy as np

from test import  test
from train import training_val

import logging

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

from utils import  get_args


def get_cmdline_args_and_run():
    args = get_args(LOGS_PATH, CHK_POINT)
    batch_size = args.batchsize
    n_epochs = args.nepochs

    possible_execution_modes = ['train', 'val', 'test']
    if not args.mode in possible_execution_modes:
        print(
            'Error: Execution mode {} is unknown. Please choose one of {}'.format(args.mode, possible_execution_modes))

    if not args.seed == -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.logdir):
        print("Creating log directory {}".format(args.logdir))
        os.makedirs(args.logdir)

    ckpt = CHK_POINT if args.mode == "train" else TEST_CHK_POINT
    ckpt = ckpt if ckpt else None

    print('load', ckpt)

    if args.mode == 'train' or args.mode == 'val':
        training_val(args,
                     json_path=os.path.join(JSON_FILE_PATH, MIDOG_JSON_FILE),
                     slide_folder=MIDOG_DEFAULT_PATH,
                     train_ids=TRAINING_IDS,
                     batch_size=batch_size,
                     n_epochs=n_epochs,
                     check_point_path=CHK_POINT_PATH,
                     is_cluster=True,
                     ckpt=ckpt)

    if args.mode == 'test':
        test(args, os.path.join(JSON_FILE_PATH, MIDOG_JSON_FILE),MIDOG_DEFAULT_PATH, TEST_IDS, ckpt)


if __name__ == "__main__":
    get_cmdline_args_and_run()

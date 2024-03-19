import albumentations as A
import torch

import time
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import viz_utils
from midog_dataset import SlideContainer
from network import MyRetinaModel
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from sklearn.model_selection import train_test_split
from midog_dataset import MIDOGTrainDataset
from utils import  skewed_sample_fn, rand_sample_fn, get_annotation_data
from dpdl_defaults_cluster import prob_threshold


def training_val(args,
                 json_path= '',
                 slide_folder ='',
                 train_ids = [],
                 batch_size = 8,
                 n_epochs =200,
                 check_point_path='',
                 is_cluster = False,
                 ckpt = None):

    batch_size = batch_size
    n_epochs = n_epochs

    print('Training on is_cluster', is_cluster)

    annotation_data = get_annotation_data(json_path)

    # TODO
    list_image_filenames = viz_utils.filter_files(train_ids, annotation_data)
    print(f'train on {len(list_image_filenames)} against {len(train_ids)}')
    # random.shuffle(list_image_filenames)

    # TODO: Normalization
    tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, p=0.5),
        # norm_fn(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Note: we only get the mitotic figures, not the impostors
    categories = [1]
    res_level = 0

    containers = []
    for image_filename in list_image_filenames:
        # filter annotations by image_id and desired annotation type
        bboxes, labels = viz_utils.get_bboxes(image_filename, annotation_data, categories)

        image_id = viz_utils.image_filename2id(image_filename)
        containers.append(SlideContainer(os.path.join(slide_folder, image_filename), image_id, y=[bboxes, labels],
                                         level=res_level, width=args.patchsize, height=args.patchsize))

    # split the train set into train and validation
    train_containers, val_containers = train_test_split(containers, train_size=0.8, random_state=args.seed)

    dataset = MIDOGTrainDataset(train_containers, patches_per_slide=args.npatchtrain, transform=tfms,
                                sample_func=skewed_sample_fn)

    val_dataset = MIDOGTrainDataset(val_containers, patches_per_slide=args.npatchval, transform=tfms,
                                    sample_func=rand_sample_fn)

    # this is not ideal but use num_workers=0 - there seems to be an bug in openslide that causes missed pixels during multi-threading
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=viz_utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=viz_utils.collate_fn)

    print("Cuda available: {}".format(torch.cuda.is_available()), flush=True)
    # Initialize a trainer
    cur_time = time.time()
    time_str = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=check_point_path)
    call_backs =  [lr_monitor, checkpoint_callback]

    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=args.logdir, version='version_lr{lr}_p{p}_b{b}_{t}'.format(
            lr=args.learningrate, p=args.patchsize, b=batch_size, t=time_str)),
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=n_epochs,
        log_every_n_steps=5,
        callbacks=call_backs
    )
    my_detection_model = MyRetinaModel(num_classes=2, iterations_epoch=len(data_loader), lr=args.learningrate,
                                       epochs=n_epochs, detectthresh_val= prob_threshold)
    # Train the model

    print('load', ckpt)

    if args.checkptfile:
        my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)
    if args.mode == 'train':
        trainer.fit(my_detection_model, data_loader, val_data_loader, ckpt_path=ckpt)
    if args.mode == 'val':
        print('validation', len(val_data_loader))
        trainer.validate(my_detection_model, val_data_loader, ckpt_path=ckpt)
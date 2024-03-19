from utils import   get_annotation_data
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import viz_utils
from midog_dataset import SlideContainer
from midog_dataset import MIDOGTestDataset
from network import MyRetinaModel
from dpdl_defaults_cluster import  prob_threshold

import evaluation
import nms
import os

def test(args, json_path, slide_folder, test_ids, ckpt):
    annotation_data = get_annotation_data(json_path)

    test_images = viz_utils.filter_files(test_ids, annotation_data)
    print(f'test on {len(test_images)} againits {len(test_ids)}', test_images[0])

    test_batchsize = 1
    categories = [1]
    res_level = 0

    all_predictions = []
    all_gt = []

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        logger=None,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.nepochs,
        log_every_n_steps=5,
        callbacks=[lr_monitor]
    )

    for image_filename in test_images:
        # get the corresponding container and dataset
        image_id = viz_utils.image_filename2id(image_filename)

        bboxes, labels = viz_utils.get_bboxes(image_filename, annotation_data, categories)
        container = SlideContainer(os.path.join(slide_folder, image_filename), image_id, y=[bboxes, labels],
                                   level=res_level, width=args.patchsize, height=args.patchsize)

        cur_dataset = MIDOGTestDataset(container)
        cur_test_dataloader = torch.utils.data.DataLoader(cur_dataset, batch_size=test_batchsize,
                                                          shuffle=False,  # important!
                                                          num_workers=0, collate_fn=viz_utils.collate_fn)

        my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)
        prediction = trainer.predict(model=my_detection_model, dataloaders=cur_test_dataloader)
        image_pred = torch.empty((0, 4))
        image_scores_raw = torch.empty((0))

        for batch_id, pred_batch in enumerate(prediction):
            for image_id, pred in enumerate(pred_batch[1]):
                # predictions comes from patch - we need the corresponding coordinate in the whole WSI image
                cur_global_pred = cur_dataset.local_to_global(batch_id * test_batchsize + image_id, pred['boxes'])
                image_pred = torch.cat([image_pred, cur_global_pred])
                image_scores_raw = torch.cat([image_scores_raw, pred["scores"]])

        # Minimum score in accepting the prediction
        image_pred_th = image_pred[image_scores_raw > prob_threshold]
        scores  = image_scores_raw[image_scores_raw > prob_threshold]

        image_pred_cthw = viz_utils.tlbr2cthw(image_pred_th)[:, :2]

        # Remove overly overlapping patches here
        image_pred_cthw = nms.nms(image_pred_cthw, scores, prob_threshold)
        image_gt_cthw = viz_utils.tlbr2cthw(cur_dataset.get_slide_labels_as_dict()['boxes'])[:, :2]

        all_predictions.append(image_pred_cthw)

        all_gt.append(image_gt_cthw)


    # Evaluation on test set
    tp, fp, fn = evaluation.get_confusion_matrix(all_gt, all_predictions)
    aggregates = evaluation.get_metrics(tp, fp, fn)
    print("The performance on the test set for the current setting was \n" +
          "F1-score:  {:.3f}\n".format(aggregates["f1_score"]) +
          "Precision: {:.3f}\n".format(aggregates["precision"]) +
          "Recall:    {:.3f}\n".format(aggregates["recall"]))

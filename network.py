import evaluation
import viz_utils
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetHead
import torch.optim
from typing import Any, Callable, Dict, List, Optional, Tuple, OrderedDict


class MyRetinaModel(LightningModule):
    
    def __init__(self, num_classes=2, iterations_epoch=100, lr=1e-4, epochs=200, detectthresh_val=0.5) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.iterations_epoch = iterations_epoch
        self.detectthresh_val = detectthresh_val

        self.sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [4, 8, 16, 32, 64])
        self.ratios = ((1.0,),) * len(self.sizes)
        
        # load a model pre-trained on COCO
        self.model = retinanet_resnet50_fpn_v2(weights='DEFAULT')

        # replace the pre-trained head with a new one and set a new anchor generator
        self.model.anchor_generator = AnchorGenerator(sizes=self.sizes, aspect_ratios=self.ratios)
        self.model.head = RetinaNetHead(self.model.backbone.out_channels, self.model.anchor_generator.num_anchors_per_location()[0], self.num_classes)
        self.val_step_outputs = []

    def get_RetinaNet_validation_loss(self, images, targets):

        images, targets = self.model.transform(images, targets)

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, torch.Tensor]] = []
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            # compute the losses
            losses = self.model.compute_loss(targets, head_outputs, anchors)
        
        return losses

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # using PyTorch Lighting logging
        self.log("train_loss", losses)
        return losses

    def on_validation_epoch_end(self):
        conf_mat = torch.sum(torch.stack([v[2] for v in self.val_step_outputs]), dim=0)
        binary_metrics = evaluation.get_metrics(*conf_mat)
        self.log("val_f1", binary_metrics["f1_score"])
        self.log("val_precision", binary_metrics["precision"])
        self.log("val_recall", binary_metrics["recall"])
        self.val_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images, targets)
        loss_dict = self.get_RetinaNet_validation_loss(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        bboxes_cat = torch.cat([t["boxes"] for t in targets])
        predictions_cat = torch.cat([p["boxes"][predictions[i]["scores"]>self.detectthresh_val] for i, p in enumerate(predictions)])
        boxes_cthw = viz_utils.tlbr2cthw(bboxes_cat)
        predictions_cthw = viz_utils.tlbr2cthw(predictions_cat)

        tp, fp, fn = evaluation.get_confusion_matrix(boxes_cthw[:, :2].cpu(), predictions_cthw[:, :2].cpu())

        # using PyTorch Lighting logging
        self.log("val_loss", losses)
        self.val_step_outputs.append([predictions, losses, torch.Tensor([tp, fp, fn])])
        return predictions, losses, torch.Tensor([tp, fp, fn])
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        prediction = self.model(images)
        loss_dict = self.get_RetinaNet_validation_loss(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        self.log("test_loss", losses)
        return {'test_loss': losses, 'preds': prediction, 'target': targets}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        return x, self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.epochs,
                                                                  steps_per_epoch=self.iterations_epoch)
        return [optimizer], [cyclic_lr_scheduler]

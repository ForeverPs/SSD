import torch
from utils import Loss
from resnet import resnet50
from torch import nn, Tensor
from torch.jit.annotations import List
from utils import dboxes300, Encoder, PostProcess


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # change stride of conv4_block1 from 2 to 1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes
        # number of default bounding boxes in each feature map
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)

        # output of location regression and classification
        location_extractors = list()
        confidence_extractors = list()

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        # location regression layers and classification layers
        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        # all default bounding boxes in SSD
        # shape [8732, 4]
        default_box = dboxes300()
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def _build_additional_features(self, input_size):
        # feature map 2 to feature map 6
        additional_blocks = list()
        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs, confs = list(), list()

        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # results in each feature map
            # [batch, 4n, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*num_classes, feat_size, feat_size] -> [batch, num_classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))
        # concatenate the results of all feature map
        # shape [batch, 4, 8732], [batch, num_classes, 8732]
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        # feature map 1 : 38x38x1024
        x = self.feature_extractor(image)

        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.additional_blocks:
            # feature map 2 to feature map 6
            # shape : 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
            x = layer(x)
            detection_features.append(x)

        # locs [batch, 4, 8732]
        # confs [batch, num_classes, 8732]
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:
            # Target
            # bboxes_out [8732, 4]
            # labels_out [8732]
            bboxes_out = targets['boxes']
            # swap dimension
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = targets['labels']

            # locs : predicted location [batch, 4, 8732]
            # confs : predicted object label [batch, 8732]
            # bboxes_out : target location [batch, 4, 8732]
            # labels_out : target labels [batch, 8732]
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {'total_losses': loss}
        results = self.postprocess(locs, confs)
        return results


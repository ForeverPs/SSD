import torch
import itertools
import numpy as np
from math import sqrt
from torch import nn, Tensor
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List


class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # totally empirical scale factor
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        # (xc, yc, w, h) : [8732, 4] -> [4, 8732] -> [1, 4, 8732]
        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

        # location regression loss and classification loss
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        # loc : predicted results of SSD300
        # dboxes : default bounding boxes
        # convert loc to (gx, gy, gw, gh)
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        # ploc : predicted location [batch, 4, 8732]
        # plabel : predicted label [batch, num_classes, 8732]
        # gloc : ground truth location [batch, 4, 8732]
        # glabel : ground truth label [batch, num_classes, 8732]

        # index of positive samples
        mask = torch.gt(glabel, 0)
        # number of positive samples
        pos_num = mask.sum(dim=1)

        # convert ground truth location to (gx, y, gw, gh), shape [batch, 4, 8732]
        # (gx', gy', gw', gh') is the output of SSD
        vec_gd = self._location_vec(gloc)

        # location loss of positive samples
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)

        # Hard Negative Mining
        # total classification loss
        con = self.confidence_loss(plabel, glabel)

        # index of top-k negative samples
        # k = 3 * pos_num
        con_neg = con.clone()
        con_neg[mask] = 0.0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)

        # classification loss of part negative samples and total positive samples
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        # ignore the image without objects
        total_loss = loc_loss + con_loss
        num_mask = torch.gt(pos_num, 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        positive_image_loss_mean = (total_loss * num_mask / pos_num).mean(dim=0)
        return positive_image_loss_mean


def box_area(boxes):
    # boxes [batch, 4] in (xmin, ymin, xmax, ymax)
    # return [batch]
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    # boxes1 [N, 4] in (xmin, ymin, xmax, ymax)
    # boxes2 [M, 4] in (xmin, ymin, xmax, ymax)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt : left-top point of intersection [N, M, 2]
    # rb : right-bottom point of intersection [N, M, 2]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # w and h of intersection
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class Encoder(object):
    """
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        # the number of default bounding boxes
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    # IoU greater than 0.5
    def encode(self, bboxes_in, labels_in, criteria=0.5):
        # bboxes_in : ground truth bounding boxes in (xmin, ymin, xmax, ymax) with shape [n_objects, 4]
        # labels_in : ground truth labels with shape [n_objects]
        # criteria : IoU threshold of bounding boxes
        # bboxes_out : [8732, 4]
        # labels_out : [8732]

        # bboxes_in : [n_objects, 4]
        # dboxes : [8732, 4]
        # ious : [n_objects, 8732]
        ious = calc_iou_tensor(bboxes_in, self.dboxes)

        # best match of each default bounding boxes to ground truth bounding boxes
        # shape : [8732]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)

        # best match of each ground truth bounding boxes to default bounding boxes
        # shape : [n_objects]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best iou 2.0 (greater than criteria)
        # set the default bounding boxes that has the max IoU with ground truth as positive samples
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        # symmetric transform for best match pairs (default bounding boxes, ground truth bounding boxes)
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # index of default bounding boxes whose IoU with ground truth is greater than criteria
        masks = best_dbox_ious > criteria

        # nboxes : number of default bounding boxes
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        bboxes_out = self.dboxes.clone()
        # change the positive default bounding boxes into ground truth bounding boxes
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # transform format to (xc, yc, w, h)
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # xc
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # yc
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        # bboxes_out : transformed from ground truth bounding boxes, shape [8732, 4], (xc, yc, w, h)
        # labels_out : transformed from ground truth labels, shape [8732]
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        # bboxes_in : predicted bounding box results of SSD, shape [batch, 4, 8732], raw, (gx, gy, gw, gh)
        # scores_in : predicted label results of SSD, shape [batch, num_classes, 8732], raw, without softmax mapping
        if bboxes_in.device == torch.device('cpu'):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # change size
        # bboxes_in : [batch, 8732, 4]
        # scores_in : [batch, 8732, num_classes]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        # bboxes_in : (gx, gy, gw, gh)
        # convert to (cx, cy, w, h)
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # convert to (xmin, ymin, xmax, ymax)
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        # return detected coordinates (xmin, ymin, xmax, ymax) of bounding boxes and corresponding scores
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        # bboxes_in : predicted bounding box results of SSD, shape [batch, 4, 8732], raw, (gx, gy, gw, gh)
        # scores_in : predicted label results of SSD, shape [batch, num_classes, 8732], raw, without softmax mapping
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = list()
        # tensor.split(num_chunk, dim)
        # split bboxes along the first dimension, each chunk has dimension 1
        # bboxes : [batch, 8732, 4]
        # probs : [batch, 8732, num_classes]
        # bbox : [1, 8732, 4], indicates the result of a single image
        # prob : [1, 8732, num_classes], indicates the result of a single image
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        # bboxes_in : [8732, 4]
        # scores_in : [8732, num_classes]

        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # the min-max of bboxes should be 0-1
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # index 0 is background
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only top-k scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x num_classes)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
            max_num : the greatest number of bounding boxes with score greater than 0.05
        """

        bboxes_out = list()
        scores_out = list()
        labels_out = list()

        # nms
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            # [8732, 1] -> [8732]
            score = score.squeeze(1)

            # ignore the object with score smaller than 0.05
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            # sort the result from min to max
            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = list()

            # tensor.numel() : the number of elements on tensor
            # score_idx_sorted.numel() : the number of objects whose scores are greater than 0.05
            while score_idx_sorted.numel() > 0:
                # the gretest score
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]

                # bounding box () with greatest score
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)

                # IoU between bounding boxes whose score greater than 0.05 and the bounding box with greatest score
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # ignore the bounding boxes whose IoU with the best bounding box is greater than criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]

                # store the best bounding boxes
                candidates.append(idx)

            # record all candidates survived from nms
            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        # if no objects detected
        if not bboxes_out:
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        # concatenate the result list and convert to torch.tensor
        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.as_tensor(labels_out, dtype=torch.long)

        # sort all detected objects (after nms), and return the top-max_output
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        # fig_size = 300 in SSD300
        self.fig_size = fig_size

        # size of feature map in SSD300
        # from feature map 1 to feature map 6 : [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size

        # empirically scale factor
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # receptive field of element in each feature map
        # SSD300 : [8, 16, 32, 64, 100, 300]
        self.steps = steps

        # size of default bounding boxes in each feature map
        # SSD300 : [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales

        # relative
        fk = fig_size / np.array(steps)

        # w / h of bounding boxes in each feature map
        # SSD300 : [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios

        self.default_boxes = list()
        for idx, sfeat in enumerate(self.feat_size):
            # relative ratio
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)

            # two default bounding boxes with ratio 1: 1
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # default bounding boxes with other ratio
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # all bounding boxes in this feature map
            for w, h in all_sizes:
                # (i, j) = (range(k), range(k))
                # k = size of this feature map
                for i, j in itertools.product(range(sfeat), repeat=2):
                    # centered coordinates of bounding boxes
                    # cx * fig_size : x coordinate of bounding box in original image
                    # cy * fig_size : y coordinate of bounding box in original image
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)

        # Other format : (xmin, ymin, xmax, ymax)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, :2] = self.dboxes[:, :2] - 0.5 * self.dboxes[:, 2:]   # xmin, ymin
        self.dboxes_ltrb[:, 2:] = self.dboxes[:, :2] + 0.5 * self.dboxes[:, 2:]   # xmax, ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # (xmin, ymin, xmax, ymax)
        if order == 'ltrb':
            return self.dboxes_ltrb
        # (cx, cy, w, h)
        if order == 'xywh':
            return self.dboxes


def dboxes300():
    # all default bounding boxes in SSD300
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def nms(boxes, scores, iou_threshold):
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


# Unknown
def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [8732, 4] -> [1, 8732, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0), requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Raw SSD Output
            bboxes_in: [batch, 4, 8732], (gx, gy, gw, gh)
            scores_in: [batch, num_classes, 8732]
        """

        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)

        # [batch, num_classes, 8732] -> [batch, 8732, num_classes]
        scores_in = scores_in.permute(0, 2, 1)

        # (gx, gy, gw, gh) to (xc, yc, w, h)
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # (xc, yc, w, h) to (xmin, ymin, xmax, ymax)
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]
        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # relative coordinates of bboxes in [0, 1]
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

        # remove low scoring boxes. self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes (small enough)
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        keep = keep.nonzero().squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only top-k scoring predictions
        # k = num_output
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]
        return bboxes_out, labels_out, scores_out

    def forward(self, bboxes_in, scores_in):
        # bboxes_in : [batch, 8732, 4] in (gx, gy, gh, gw)
        # scores_in : [batch, 8732, num_classes]
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # batch images :
        # bboxes: [batch, 8732, 4]
        # probs : [batch, 8732, num_classes]

        # single image :
        # bbox : [1, 8732, 4]
        # prob : [1, 8732, num_classes]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs

import os
import torch
from PIL import Image
from lxml import etree
from transform import ssd_transform
from torch.utils.data import Dataset, DataLoader


def assign_class_label():
    classes = ['background', 'aeroplane', 'bicycle', 'bird',
               'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']
    label_dict = {}
    for i in range(len(classes)):
        label_dict[classes[i]] = i
    return label_dict


class VOC2012DataSet(Dataset):
    # voc_root = ./dataset/VOC2012/
    def __init__(self, voc_root, transforms, train_set='train.txt'):
        self.root = voc_root
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')
        txt_list = os.path.join(self.root, 'ImageSets', 'Main', train_set)

        # all xml annotations in train_set
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in read.readlines()]

        self.class_dict = assign_class_label()

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])
        data_width = int(data['size']['width'])
        height_width = [data_height, data_width]
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        boxes = list()
        labels = list()
        iscrowd = list()
        for obj in data['object']:
            # ground truth
            # relative coordinates
            xmin = float(obj['bndbox']['xmin']) / data_width
            xmax = float(obj['bndbox']['xmax']) / data_width
            ymin = float(obj['bndbox']['ymin']) / data_height
            ymax = float(obj['bndbox']['ymax']) / data_height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))

        # convert to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['height_width'] = height_width

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # def get_height_and_width(self, idx):
    #     # read xml
    #     xml_path = self.xml_list[idx]
    #     with open(xml_path) as fid:
    #         xml_str = fid.read()
    #     xml = etree.fromstring(xml_str)
    #     data = self.parse_xml_to_dict(xml)['annotation']
    #     data_height = int(data['size']['height'])
    #     data_width = int(data['size']['width'])
    #     return data_height, data_width

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        return images, targets


if __name__ == '__main__':
    batch_size = 300
    voc_root = './dataset/VOC2012/'
    transforms = ssd_transform()
    train_dataset = VOC2012DataSet(voc_root, transforms['train'], train_set='train.txt')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=8,
                                   collate_fn=train_dataset.collate_fn,
                                   drop_last=True)

    for images, targets in train_data_loader:
        batch_images = torch.stack(images, dim=0)
        batch_boxes = torch.stack([targets[i]['boxes'] for i in range(batch_size)], dim=0)
        batch_labels = torch.stack([targets[i]['labels'] for i in range(batch_size)], dim=0)
        print(batch_images.shape, batch_boxes.shape, batch_labels.shape)



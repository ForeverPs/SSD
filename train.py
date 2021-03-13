import os
import torch
from data import VOC2012DataSet
from model import SSD300, Backbone
from transform import ssd_transform
from torch.utils.data import DataLoader


def create_model(num_classes=21, device=torch.device('cpu')):
    backbone = Backbone(pretrain_path='./pretrain/resnet50.pth')
    model = SSD300(backbone=backbone, num_classes=num_classes)

    pre_ssd_path = './pretrain/nvidia_ssdpyt_fp32.pt'
    pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    pre_weights_dict = pre_model_dict['model']

    # only use the pre_trained bounding boxes regression weights
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split('.')
        if 'conf' in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    # if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    #     print('missing_keys: ', missing_keys)
    #     print('unexpected_keys: ', unexpected_keys)
    return model


def train_ssd(num_classes=21, batch_size=3, epochs=30, data_root='./dataset/VOC2012/'):
    device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes)
    model.to(device)
    data_transform = ssd_transform()

    train_dataset = VOC2012DataSet(data_root, data_transform['train'], train_set='train.txt')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4,
                                   collate_fn=train_dataset.collate_fn,
                                   drop_last=True)

    val_dataset = VOC2012DataSet(data_root, data_transform['val'], train_set='val.txt')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, collate_fn=train_dataset.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0005)

    model.train()
    print('SSD300 Training Start...')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, targets in train_data_loader:
            batch_images = torch.stack(images, dim=0)
            batch_boxes = torch.stack([targets[i]['boxes'] for i in range(batch_size)], dim=0)
            batch_labels = torch.stack([targets[i]['labels'] for i in range(batch_size)], dim=0)
            batch_image_id = torch.as_tensor([targets[i]['image_id'] for i in range(batch_size)])
            # print(batch_images.shape, batch_boxes.shape, batch_labels.shape)

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_boxes = batch_boxes.to(device)
            batch_image_id.to(device)
            batch_targets = {'boxes': batch_boxes, 'labels': batch_labels, 'image_id': batch_image_id}

            loss = model(batch_images, batch_targets)['total_losses']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('EPOCH : %03d  Training Loss : %2.4f' % (epoch, epoch_loss / len(train_data_loader)))
        torch.save(model, './saved_models/ssd300_epoch_%03d.pth' % epoch)


if __name__ == '__main__':
    train_ssd()

import os
import logging
from tkinter import Image
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

import config
from models import WSDAN
from datasets import get_test_datasets
from utils import batch_augment

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# visualize
visualize = config.visualize
savepath = config.eval_savepath
if visualize:
    os.makedirs(savepath, exist_ok=True)

# label dict
label_dict = {}
with open('./CarType_test/class.txt', 'r') as f:
    for line in f.readlines():
        id, label_name = line.strip().split(' ')
        label_dict[id] = label_name

# path
root_path = './CarType_test'
image_path = root_path + '/images'
result_path = './split-result'

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def split_list_with_batchsize(image_list, batchsize):
    reshape_list = []
    for i in range(0, len(image_list), batchsize):
        reshape_list.append(image_list[i:i+batchsize])
    return reshape_list


def main():
    try:
        ckpt = config.eval_ckpt
    except:
        logging.info('Set ckpt for evalution in config.py')
        return

    # Dataset for testing
    test_dataset = get_test_datasets(config.tag, resize=config.image_size)
    all_image_list = test_dataset.get_all_image_path()
    image_res_list = split_list_with_batchsize(all_image_list, config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    # load model
    net = WSDAN(num_classes=test_dataset.num_classes, M=config.num_attentions, net=config.net)
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    # use cuda
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.eval()
    total = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, y) in enumerate(test_loader):
            data = (X, y)
            img, label = data[0].cuda(), data[1].cuda()
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

            y_pred_crop, _, _ = net(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            _, concat_predict = torch.max(y_pred,dim=1)

            for j in range(len(X)):
                predict_la = concat_predict.data[j]
                true_la = label.data[j]
                predict_la_name = label_dict[str(predict_la.item())]
                #true_la_name = label_dict[str(true_la.item())]
                picture_name = image_res_list[i][j]
                im_path = os.path.join(image_path, picture_name)
                shutil.copy(im_path, os.path.join(result_path, predict_la_name))
                total += 1


            if visualize:
                # reshape attention maps
                attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * config.batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))

    print('finishing spliting!')


if __name__ == '__main__':
    main()
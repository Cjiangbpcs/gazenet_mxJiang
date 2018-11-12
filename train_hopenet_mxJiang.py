#!/usr/bin/env python
# coding: utf-8

# convert PyTorch hopenet scripts to MXNet

from mxboard import SummaryWriter
import sys, os, argparse, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.autograd import record
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
import mxnet.gluon.data.vision
import datasets_mxJiang, hopenet_mxJiang
import mxnet.gluon.model_zoo as model_zoo
import pickle as pk
from mxnet import ndarray as nd

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
    
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def reg_criterion(A, B):
    mse = ((A - B)**2).mean()
    return mse

if __name__ == '__main__':
    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    lr = args.lr

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet_mxJiang.Hopenet_mxJiang(mx.gluon.model_zoo.vision.resnet.BottleneckV1, [3, 4, 6, 3], 66)

    # snapshot is not required
    #model.load_parameters('../resnet50-0000.params', ctx, allow_missing=False, ignore_extra=False)
    
    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(240),
            transforms.RandomResizedCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                      
    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets_mxJiang.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets_mxJiang.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets_mxJiang.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets_mxJiang.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets_mxJiang.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets_mxJiang.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets_mxJiang.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets_mxJiang.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = gluon.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    alpha = args.alpha
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.float32(idx_tensor)
    optimizer = gluon.Trainer(model.collect_params(),'adam', {'learning_rate': lr})

    print('Ready to train network.')
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        n = 0
        for images, labels, cont_labels in train_loader:
            n += 1
            # Binned labels
            label_yaw = labels[:,0]
            label_pitch = labels[:,1]
            label_roll = labels[:,2]

            # Continuous labels
            label_yaw_cont = cont_labels[:,0]
            label_pitch_cont = cont_labels[:,1]
            label_roll_cont = cont_labels[:,2]
            if n >= 0:
                with mx.autograd.record():
                    # Forward pass
                    yaw, pitch, roll = model(images)

                    # Cross entropy loss
                    loss_yaw = criterion(yaw, label_yaw)
                    loss_pitch = criterion(pitch, label_pitch)
                    loss_roll = criterion(roll, label_roll)

                    # MSE loss
                    yaw_predicted = softmax(yaw.asnumpy())
                    pitch_predicted = softmax(pitch.asnumpy())
                    roll_predicted = softmax(roll.asnumpy())

                    yaw_predicted = np.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                    pitch_predicted = np.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                    roll_predicted = np.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                    loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont.asnumpy())
                    loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont.asnumpy())
                    loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont.asnumpy())

                    # Total loss
                    loss_yaw = loss_yaw + alpha * loss_reg_yaw
                    loss_pitch = loss_pitch + alpha * loss_reg_pitch
                    loss_roll = loss_roll + alpha * loss_reg_roll

                    loss_seq = [loss_yaw, loss_pitch, loss_roll]
                    grad_seq = [mx.nd.array((1,)) for _ in range(len(loss_seq))]
                    
                #print('loss_seq', loss_seq)
                mx.autograd.backward(loss_seq, grad_seq, retain_graph=False)
                optimizer.step(batch_size)
                      
        # Save models at numbered epochs.
        #if epoch % 1 == 0 and epoch < num_epochs:
        #    print('Taking snapshot...')
        #    model.save_parameters(
        #    'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.json')


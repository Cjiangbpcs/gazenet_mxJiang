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
import datasets_mxJiang, hopenet_mxJiang_yaw, hopenet_mxJiang_pitch, hopenet_mxJiang_roll
import datasets, hopenet
import mxnet.gluon.model_zoo as model_zoo
import pickle as pk
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock
import datetime

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
    #ctx = mx.cpu()
    ctx = mx.gpu(gpu)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')
    
    model_yaw = hopenet_mxJiang_yaw.Hopenet_mxJiang_yaw(66,ctx=ctx)
    model_pitch = hopenet_mxJiang_pitch.Hopenet_mxJiang_pitch(66,ctx=ctx)
    model_roll = hopenet_mxJiang_roll.Hopenet_mxJiang_roll(66,ctx=ctx)
    #x = mx.sym.var('data')
    #sym = model_yaw.net(x)
    #mx.viz.plot_network(sym[0])
    
    # ResNet50 structure
    model_yaw.hybridize()
    model_pitch.hybridize()
    model_roll.hybridize()
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
                                               num_workers=1)
    
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    #reg_criterion = gluon.loss.L2Loss()
    alpha = args.alpha
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.float32(idx_tensor)
    lst = list(model_yaw.collect_params().values()) + list(model_pitch.collect_params().values()) + list(model_roll.collect_params().values())
    optimizer = gluon.Trainer(lst,'sgd', {'learning_rate': lr})
    
    unique_id = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.','-')
    with SummaryWriter(logdir='./logs_gluon/logs-'+unique_id, flush_secs=5) as sw:
        try:
            global_step = 0
            for epoch in range(num_epochs):
                thefile = open('test'+'-'+str(epoch)+'.txt', 'w')
                for batch_idx, (images, labels, cont_labels) in enumerate(train_loader):
                    global_step += 1
                    images = images.as_in_context(ctx)
                    labels.as_in_context(ctx)
                    cont_labels = cont_labels.as_in_context(ctx)
                    # Forward pass
                    yaw = model_yaw(images)
                    pitch = model_pitch(images)
                    roll = model_roll(images)
                    # Binned labels
                    label_yaw = labels[:,0].as_in_context(ctx)
                    label_pitch = labels[:,1].as_in_context(ctx)
                    label_roll = labels[:,2].as_in_context(ctx)
                    # Continuous labels
                    label_yaw_cont = cont_labels[:,0].as_in_context(ctx)
                    label_pitch_cont = cont_labels[:,1].as_in_context(ctx)
                    label_roll_cont = cont_labels[:,2].as_in_context(ctx)
                    yaw.attach_grad()
                    pitch.attach_grad()
                    roll.attach_grad()
                    with mx.autograd.record():
                        # Cross entropy loss
                        loss_yaw = criterion(yaw, label_yaw).mean()
                        loss_pitch = criterion(pitch, label_pitch).mean()
                        loss_roll = criterion(roll, label_roll).mean()
                        # MSE loss
                        yaw_predicted = softmax(yaw.asnumpy(),axis=1)
                        pitch_predicted = softmax(pitch.asnumpy(),axis=1)
                        roll_predicted = softmax(roll.asnumpy(),axis=1)
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
                        loss_seq = [loss_yaw.as_in_context(ctx), loss_pitch.as_in_context(ctx), loss_roll.as_in_context(ctx)]
                        if epoch == 0 and batch_idx == 0:
                            sw.add_graph(model_yaw)
                    if global_step%10 == 0:
                        sw.add_scalar(tag='Log10_of_loss_yaw',value=loss_yaw.log10().asscalar(), global_step=global_step)
                        sw.add_scalar(tag='epoch', value=epoch, global_step=global_step)             
                    mx.autograd.backward(loss_seq)
                    optimizer.step(batch_size,ignore_stale_grad=True)    
                thefile.write("Epoch: %d; Batch %d; Loss Yaw %f; Loss Pitch %f; Loss Roll %f \n" % (epoch, batch_idx, loss_yaw.asnumpy(), loss_pitch.asnumpy(), loss_roll.asnumpy()))
                print ("Epoch: %d; Batch %d; Loss Yaw %f; Loss Pitch %f; Loss Roll %f" % (epoch, batch_idx, loss_yaw.asnumpy(), loss_pitch.asnumpy(), loss_roll.asnumpy()))           
                thefile.close()
                model_yaw.export('yawparams-'+unique_id+str(epoch))
                model_pitch.export('pitchparams-'+unique_id+str(epoch))
                model_roll.export('rollparams-'+unique_id+str(epoch))
                sw.close()
        except KeyboardInterrupt:
            print("KeyboardInterrupted")
            sw.close()
            exit(0)
            pass
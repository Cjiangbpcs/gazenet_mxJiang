# convert PyTorch gazenet scripts to MXNet

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
import datasets_mxJiang, gazenet_mxJiang_yaw_zoo, gazenet_mxJiang_pitch_zoo,gazenet_mxJiang_roll_zoo
import mxnet.gluon.model_zoo as model_zoo
import pickle as pk
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock
import datetime
import copy
from mxnet.gluon.loss import Loss
import time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the gazenet network.')
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
    
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def reg_criterion(A, B):
    mse = ((A - B)**2)
    return mse

class compute_cost(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(compute_cost, self).__init__(weight, batch_axis, **kwargs)
    
    def hybrid_forward(self, F, yaw, label_yaw, label_yaw_cont, idx_tensor):
        loss_yaw = criterion(yaw, label_yaw)
        yaw_predicted = softmax(yaw.asnumpy(),axis=1)
        #print('loss_entropy',loss_yaw)
        yaw_predicted = np.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        loss_reg_yaw = nd.array(reg_criterion(yaw_predicted, label_yaw_cont.asnumpy())).as_in_context(ctx)
        #print('loss_reg',loss_reg_yaw)
        total_loss_yaw = loss_yaw + alpha * loss_reg_yaw
        #print('total_loss', total_loss_yaw)
        return yaw_predicted, total_loss_yaw

if __name__ == '__main__':
    
    mx.random.seed(int(time.time()))
    
    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    lr = args.lr
    lr_period = 15
    lr_decay = 0.1
    ctx = mx.gpu(gpu)
    
    model_yaw = gazenet_mxJiang_yaw_zoo.Gazenet_mxJiang_yaw_zoo(66,ctx=ctx)
    model_pitch = gazenet_mxJiang_pitch_zoo.Gazenet_mxJiang_pitch_zoo(66,ctx=ctx)
    model_roll = gazenet_mxJiang_roll_zoo.Gazenet_mxJiang_roll_zoo(66,ctx=ctx)
    
    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(240),
                        #transforms.RandomResizedCrop(224), 
                                transforms.ToTensor(),
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
    criterion_reg = gluon.loss.L2Loss()
    comp_cost = compute_cost()
    alpha = args.alpha
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.float32(idx_tensor)
    
    #get_ignored_params(model) lr=0
    model_yaw.collect_params('resnetv10_conv0_weight.*').setattr('lr_mult', 0)
    model_yaw.collect_params('resnetv10_batchnorm0_gamma').setattr('lr_mult', 0)
    model_yaw.collect_params('resnetv10_batchnorm0_betta').setattr('lr_mult', 0)
    
    model_pitch.collect_params('resnetv10_conv0_weight.*').setattr('lr_mult', 0)
    model_pitch.collect_params('resnetv10_batchnorm0_gamma').setattr('lr_mult', 0)
    model_pitch.collect_params('resnetv10_batchnorm0_betta').setattr('lr_mult', 0)
    
    model_roll.collect_params('resnetv10_conv0_weight.*').setattr('lr_mult', 0)
    model_roll.collect_params('resnetv10_batchnorm0_gamma').setattr('lr_mult', 0)
    model_roll.collect_params('resnetv10_batchnorm0_betta').setattr('lr_mult', 0)
    
    #get_ignored_params(model) model.eval()
    #model_yaw.collect_params('resnetv10_batchnorm0_running_mean').setattr('momentum',1)
    #model_yaw.collect_params('resnetv10_batchnorm0_running_var').setattr('momentum',1)
    
    #model_pitch.collect_params('resnetv10_batchnorm0_running_mean').setattr('momentum',0.01)
    #model_pitch.collect_params('resnetv10_batchnorm0_running_var').setattr('momentum',0.01)
    
    #model_roll.collect_params('resnetv10_batchnorm0_running_mean').setattr('momentum',0.01)
    #model_roll.collect_params('resnetv10_batchnorm0_running_var').setattr('momentum',0.01)

    #get_fc_params(model)
    model_yaw.collect_params('dense.*').setattr('lr_mult', 5)
    model_pitch.collect_params('dense.*').setattr('lr_mult', 5)
    model_roll.collect_params('dense.*').setattr('lr_mult', 5)
    
    model_yaw.model_yaw.bias.set_data(mx.nd.random.uniform(-0.2,0.2,shape=(66,),ctx=ctx))
    
    optimizer_yaw = gluon.Trainer(model_yaw.collect_params(),'adam', {'learning_rate': lr})
    optimizer_pitch = gluon.Trainer(model_pitch.collect_params(),'adam', {'learning_rate': lr})
    optimizer_roll = gluon.Trainer(model_roll.collect_params(),'adam', {'learning_rate': lr})
    #print('model_yaw collect prams==> ',model_yaw.collect_params())
    
    # ResNet50 structure
    model_yaw.hybridize()
    model_pitch.hybridize()
    model_roll.hybridize()
    
    unique_id = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.','-')
    with SummaryWriter(logdir='/data/deep-head-pose/logs_gluon/logs-'+unique_id, flush_secs=5) as sw:
        try:
            global_step = 0
            for epoch in range(num_epochs):        
                thefile = open('/data/deep-head-pose/test'+'-'+str(epoch)+str(datetime.datetime.now())+'.txt', 'w')
                #if epoch > 0 and epoch % lr_period == 0:
                #    optimizer.set_learning_rate(optimizer.learning_rate * lr_decay)
                #if epoch == 20:
                #    model_yaw.net.collect_params('resnetv10_stage.*_.*').setattr('lr_mult', 0.001)
                #    model_pitch.net.collect_params('resnetv10_stage.*_.*').setattr('lr_mult', 0.001)
                #    model_roll.net.collect_params('resnetv10_stage.*_.*').setattr('lr_mult', 0.001)
                for batch_idx, (images, labels, cont_labels) in enumerate(train_loader):
                    #print('images=>',images)
                    global_step += 1
                    images = images.as_in_context(ctx)
                    labels.as_in_context(ctx)
                    cont_labels = cont_labels.as_in_context(ctx)
                    # Binned labels
                    label_yaw = labels[:,0].as_in_context(ctx)
                    label_pitch = labels[:,1].as_in_context(ctx)
                    label_roll = labels[:,2].as_in_context(ctx)
                    # Continuous labels
                    label_yaw_cont = cont_labels[:,0].as_in_context(ctx)
                    label_pitch_cont = cont_labels[:,1].as_in_context(ctx)
                    label_roll_cont = cont_labels[:,2].as_in_context(ctx)
                    yaw = nd.ones((1,66))
                    yaw.attach_grad()
                    with mx.autograd.record():
                        yaw = model_yaw(images)
                        #print('yaw',yaw.mean())
                        yaw_predicted, total_loss_yaw = comp_cost(yaw, label_yaw, label_yaw_cont, idx_tensor)
                        #print('yaw_predicted', yaw_predicted)
                        mx.autograd.backward(total_loss_yaw.nansum().as_in_context(ctx))
                        if global_step%10 == 0:
                            sw.add_scalar(tag='loss_yaw',value=total_loss_yaw.nansum().asnumpy(), global_step=global_step)
                    #print('loss_yaw', total_loss_yaw)
                    #print('fc_yaw grad==>', [w.grad() for u, w in model_yaw.model_yaw.collect_params().items()])
                    optimizer_yaw.step(batch_size)
                    #print('yaw_grad==>', yaw.grad)
                    with mx.autograd.record():
                        pitch = model_pitch(images)
                        #print('pitch',pitch)
                        pitch_predicted, total_loss_pitch = comp_cost(pitch, label_pitch, label_pitch_cont, idx_tensor)
                        #print('pitch_predicted',pitch_predicted)
                        mx.autograd.backward(total_loss_pitch.nansum().as_in_context(ctx))
                        if global_step%10 == 0:
                            sw.add_scalar(tag='loss_pitch',value=total_loss_pitch.mean().asnumpy(), global_step=global_step)
                    optimizer_pitch.step(batch_size)
                    with mx.autograd.record():
                        roll = model_roll(images)
                        #print('roll',roll)
                        roll_predicted, total_loss_roll = comp_cost(roll, label_roll, label_roll_cont, idx_tensor)
                        #print('roll_predicted',roll_predicted)
                        mx.autograd.backward(total_loss_roll.nansum().as_in_context(ctx))
                        if global_step%10 == 0:
                            sw.add_scalar(tag='loss_roll',value=total_loss_roll.mean().asnumpy(), global_step=global_step)
                            sw.add_scalar(tag='epoch', value=epoch, global_step=global_step)              
                    optimizer_roll.step(batch_size)
                    loss_yaw_sum = total_loss_yaw.mean().asscalar() 
                    loss_pitch_sum = total_loss_pitch.mean().asscalar()
                    loss_roll_sum = total_loss_roll.mean().asscalar()
                    now_lr = optimizer_yaw.learning_rate
                    thefile.write("Epoch: %d; Batch %d; Loss Yaw %f; Loss Pitch %f; Loss Roll %f \n" % (epoch, batch_idx, loss_yaw_sum, loss_pitch_sum, loss_roll_sum))
                    if batch_idx%100 == 0:
                        print ("Epoch: %d; Batch %d; Losses of Yaw: %f; Pitch: %f; Roll: %f; Now_Lr: %f" % (epoch, batch_idx, loss_yaw_sum, loss_pitch_sum, loss_roll_sum, now_lr))  
                        print('predicted==>', yaw_predicted, pitch_predicted, roll_predicted)
                thefile.close()
                model_yaw.export('/data/deep-head-pose/yawparams-'+unique_id+str(epoch))
                model_pitch.export('/data/deep-head-pose/pitchparams-'+unique_id+str(epoch))
                model_roll.export('/data/deep-head-pose/rollparams-'+unique_id+str(epoch))
            sw.close()
        except KeyboardInterrupt:
            print("KeyboardInterrupted")
            sw.close()
            exit(0)
            pass
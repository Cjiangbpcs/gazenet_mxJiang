# convert PyTorch gazenet.py to Gluon roll.

import mxnet as mx
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from mxnet.gluon import HybridBlock
from mxnet.gluon.model_zoo import vision

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.autograd import record
from mxnet.gluon import Block
import math
from mxnet import nd
import time

def freeze_bn(block):
    try:
        iter(block)
        for b in block:
            freeze_bn(b)
        return
    except TypeError:
        pass
    
    if isinstance(block, nn.BatchNorm):
        #print('freeze', block.name)
        block._kwargs['use_global_stats'] = True
    elif isinstance(block, vision.BottleneckV1):
        freeze_bn(block.body)
        freeze_bn(block.downsample)

class Gazenet_mxJiang_roll_zoo(HybridBlock):
    def __init__(self, num_bins, **kwargs):
        
        ctx = kwargs.pop('ctx')
        super(Gazenet_mxJiang_roll_zoo, self).__init__(**kwargs)
        self.net = mx.gluon.nn.HybridSequential(prefix='')
        with self.net.name_scope():
            mx.random.seed(int(time.time()))
            self.model_resnet50 = vision.resnet50_v1(pretrained=True, ctx=ctx, root='./')
            #self.model_resnet50.features[1]._kwargs['use_global_stats'] = True
            freeze_bn(self.model_resnet50.features)
            #print('net features:', self.model_resnet50.features[1])
            self.net.add(self.model_resnet50)
            self.model_roll = mx.gluon.nn.Dense(num_bins)
            self.model_roll.collect_params().initialize(mx.initializer.Uniform(1/math.sqrt(2048)), ctx=ctx, force_reinit=True)
            self.model_roll.bias.set_data(mx.nd.random.uniform(-0.2,0.2,shape=(num_bins,),ctx=ctx))
            self.net.add(self.model_roll)       
            
    def hybrid_forward(self, F, x):
        pre_roll = self.net(x)
        model_net_params = self.net.collect_params()
        #print('conv1',model_net_params['resnetv10_conv0_weight'].data())
        #print('bn1 running mean',model_net_params['resnetv10_batchnorm0_running_mean'].data())
        #print('bn1 running var',model_net_params['resnetv10_batchnorm0_running_var'].data())
        #print('bn1 gamma',model_net_params['resnetv10_batchnorm0_gamma'].data())
        #print('bn1 beta',model_net_params['resnetv10_batchnorm0_beta'].data())
        
        return pre_roll
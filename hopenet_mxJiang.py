# convert PyTorch hopenet.py to MXNet

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

class Hopenet_mxJiang(HybridBlock):
    def __init__(self, block, layers, num_bins, **kwargs):
        #ctx = mx.gpu(0)
        super(Hopenet_mxJiang, self).__init__(**kwargs)
        self.net = nn.HybridSequential(prefix='')
        with self.net.name_scope():
            sym = mx.sym.load('resnet50-symbol.json')
            new_sym = sym.get_internals()['resnetv13_dense0_fwd_output']
            self.model_resnet50 = gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data'))
            self.model_yaw = mx.gluon.nn.Dense(num_bins)
            self.model_pitch = mx.gluon.nn.Dense(num_bins)
            self.model_roll = mx.gluon.nn.Dense(num_bins)
            
            self.model_resnet50.collect_params().load('resnet50-0000.params', allow_missing=False, ignore_extra=True)
            self.model_yaw.collect_params().\
             initialize(mx.initializer.Xavier(rnd_type='uniform', factor_type='in', magnitude=0.95))
            self.model_pitch.collect_params().\
             initialize(mx.initializer.Xavier(rnd_type='uniform', factor_type='in', magnitude=0.95))
            self.model_roll.collect_params().\
             initialize(mx.initializer.Xavier(rnd_type='uniform', factor_type='in', magnitude=0.95))
            
    def hybrid_forward(self, F, x):
        x = self.model_resnet50(x)
        pre_yaw = self.model_yaw(x)
        pre_pitch = self.model_pitch(x)
        pre_roll = self.model_roll(x)
        #print("========>", pre_yaw)
        return pre_yaw, pre_pitch, pre_roll
            
        
   
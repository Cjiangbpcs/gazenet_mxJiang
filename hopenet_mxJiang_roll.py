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

class Hopenet_mxJiang_roll(HybridBlock):
    def __init__(self, num_bins, **kwargs):
        ctx = kwargs.pop('ctx')
        super(Hopenet_mxJiang_roll, self).__init__(**kwargs)
        self.net = nn.HybridSequential(prefix='hopenet')
        with self.net.name_scope():
            sym = mx.sym.load('resnet50-symbol.json')
            new_sym = sym.get_internals()['resnetv13_pool1_fwd_output']
            self.model_resnet50 = gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data'))
            self.model_extra = Hopenet_mxJiang_extra(num_bins, prefix='extra')
            self.net.add(self.model_resnet50)
            self.net.add(self.model_extra)
            
            self.model_resnet50.collect_params().load('resnet50-0000.params', ctx=ctx,allow_missing=False, ignore_extra=True)
            self.model_extra.collect_params().\
             initialize(mx.initializer.Xavier(rnd_type='uniform', factor_type='in', magnitude=0.95), ctx=ctx)
            
    def hybrid_forward(self, F, x):
        x = self.model_resnet50(x)
        #x = x.reshape((0, -1))
        pre_roll = self.model_extra(x)
        return pre_roll
        
class Hopenet_mxJiang_extra(HybridBlock):
    def __init__(self, num_bins, **kwargs):
        super(Hopenet_mxJiang_extra, self).__init__(**kwargs)
        
        self.model_roll = nn.HybridSequential(prefix='roll')
        with self.model_roll.name_scope():
            self.model_roll.add(mx.gluon.nn.Dense(num_bins))
            
    def hybrid_forward(self, F, x):  
        out_model_roll = self.model_roll(x)   
        return out_model_roll    
   
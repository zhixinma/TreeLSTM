import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn, rnn
from mxnet import autograd
from nltk.tree import Tree


class N_aryTreeLstm(nn.Block):
    def __init__(self, dim_h=300, tag_num=66, max_child_num=6, **kwargs):
        super(N_aryTreeLstm, self).__init__(**kwargs)
        with self.name_scope():
            self.dim_vec = pretrain_vec.vec_len
            self.dim_h   = dim_h
            self.word_embedding = nn.Embedding(len(pretrain_vec), self.dim_vec)
            self.tag_embeding   = nn.Embedding(tag_num, self.dim_vec)
            # input gate
            self.Wi = self.params.get('Wi', shape=(dim_h, self.dim_vec), init=mx.init.Xavier())
            self.bi = self.params.get('bi', shape=(dim_h, ), init=mx.init.Zero())
            self.Uis = [self.params.get('Ui%d'%i, shape=(dim_h, dim_h), init=mx.init.Xavier()) for i in range(max_child_num)]
            # forget gate
            self.Wf = self.params.get('Wf', shape=(dim_h, self.dim_vec), init=mx.init.Xavier())
            self.bf = self.params.get('bf', shape=(dim_h, ), init=mx.init.Zero())
            self.Ufs = [[self.params.get('Uf%d%d'%(i,j), shape=(dim_h, dim_h), init=mx.init.Xavier()) for j in range(max_child_num)] for i in range(max_child_num)]
            # output gate
            self.Wo = self.params.get('Wo', shape=(dim_h, self.dim_vec), init=mx.init.Xavier())
            self.bo = self.params.get('bo', shape=(dim_h, ), init=mx.init.Zero())
            self.Uos = [self.params.get('Uo%d'%i, shape=(dim_h, dim_h), init=mx.init.Xavier()) for i in range(max_child_num)]
            # u
            self.Wu = self.params.get('Wu', shape=(dim_h, self.dim_vec), init=mx.init.Xavier())
            self.bu = self.params.get('bu', shape=(dim_h, ), init=mx.init.Zero())
            self.Uus = [self.params.get('Uu%d'%i, shape=(dim_h, dim_h), init=mx.init.Xavier()) for i in range(max_child_num)]

    def nodeforward(self, x, cs, hs, ctx):
        x   =  nd.reshape(x, (self.dim_h,))
        _Ui =  nd.zeros((self.dim_h, ), ctx=ctx)
        _Uo =  nd.zeros((self.dim_h, ), ctx=ctx)
        _Uu =  nd.zeros((self.dim_h, ), ctx=ctx)
        _Uf = [nd.zeros((self.dim_h, ), ctx=ctx) for i in range(len(cs))]

        for idx in range(len(cs)):
            _Ui = nd.add(_Ui, nd.dot(self.Uis[idx].data(), hs[idx]))
            _Uo = nd.add(_Uo, nd.dot(self.Uos[idx].data(), hs[idx]))
            _Uu = nd.add(_Uu, nd.dot(self.Uus[idx].data(), hs[idx]))
            for j in range(len(cs)):
                _Uf[idx] = nd.add(_Uf[idx], nd.dot(self.Ufs[idx][j].data(), hs[j]))

        i =  nd.sigmoid(nd.add(nd.add(nd.dot(self.Wi.data(), x), _Ui), self.bi.data()))
        o =  nd.sigmoid(nd.add(nd.add(nd.dot(self.Wo.data(), x), _Uo), self.bo.data()))
        f = [nd.sigmoid(nd.add(nd.add(nd.dot(self.Wf.data(), x), _Uf[idx]), self.bf.data())) for idx in range(len(cs))]
        u =  nd.tanh   (nd.add(nd.add(nd.dot(self.Wu.data(), x), _Uu), self.bu.data()))

        c =  nd.zeros((self.dim_h, ), ctx=ctx)
        for idx in range(len(cs)):
            c = nd.add(c, nd.multiply(f[idx], cs[idx]))
        c = nd.add(nd.multiply(i, u), c)
        
        h = nd.multiply(o, nd.tanh(c))
        return c, h

    def forward(self, tree, inputs, idx_node, ctx):
        cs = []
        hs = []
        idx_node += 1
        if isinstance(tree, Tree):
            for child in tree:
                c_sub_q, h_sub_q = self.forward(child, inputs, idx_node, ctx)
                cs.append(c_sub_q)
                hs.append(h_sub_q)
            Input = inputs[idx_node]
        else:
            Input = inputs[idx_node]
            cs.append(Input)
            hs.append(Input)

        c_q, h_q = self.nodeforward(Input, cs, hs, ctx)
        return c_q, h_q

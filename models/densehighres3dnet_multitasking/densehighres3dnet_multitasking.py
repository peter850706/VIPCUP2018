# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer, LayerFromCallable
from niftynet.layer.gn import GNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet

"""
multitasking
Before res_05, ptv, ctv and gtv share the same weights.
After res_05, they have different weights but the same structure.
"""

class DenseHighRes3DNet(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='DenseHighRes3DNet'):

        super(DenseHighRes3DNet, self).__init__(num_classes=num_classes,
                                                w_initializer=w_initializer,
                                                w_regularizer=w_regularizer,
                                                b_initializer=b_initializer,
                                                b_regularizer=b_regularizer,
                                                acti_func=acti_func,
                                                name=name)

        self.layers = \
        [{'name': 'conv_00', 'n_features': 8, 'kernel_size': 3},
         {'name': 'ddc_01', 'downsampling_rate': 2},
         {'name': 'lat_02', 'n_features': 8, 'kernel_size': 1},
         {'name': 'res_03', 'n_features': 8, 'kernels': [3, 3], 'dilated_rates': [1, 1], 'n_blocks': 3},
         {'name': 'res_04', 'n_features': 16, 'kernels': [3, 3], 'dilated_rates': [1, 3], 'n_blocks': 3},
         {'name': 'res_05', 'n_features': 32, 'kernels': [3, 3, 3], 'dilated_rates': [1, 3, 5], 'n_blocks': 3},
         {'name': 'conv_06', 'n_features': 64, 'kernel_size': 1},
         {'name': 'conv_07', 'n_features': 2, 'kernel_size': 1},        
         {'name': 'duc_08', 'upsampling_rate': 2}]

    def layer_op(self, images, is_training=True, **unused_kwargs):
        assert (layer_util.check_spatial_dims(images, lambda x: x % 8 == 0))
        # go through self.layers, create an instance of each layer and plugin data
        layer_instances = []
        
        ### dense downsampling convoluiton
        def _reverse_subpixel(X, r):
            # rearrange voxels to slice
            def rearrangement(voxels_list, axis):
                tmp = []
                for i in range(r):
                    for j in range(i, len(voxels_list), r):
                        tmp.append(voxels_list[j])
                rearranged_voxels_list = tf.concat([voxels for voxels in tmp], axis=axis)
                return rearranged_voxels_list
            
            bsize, rows, columns, slices, channels = X.get_shape().as_list()
            X = tf.split(X, num_or_size_splits=slices, axis=3)
            X = rearrangement(X, axis=3)
            X = tf.split(X, num_or_size_splits=columns, axis=2)
            X = rearrangement(X, axis=2)
            X = tf.split(X, num_or_size_splits=rows, axis=1)
            X = rearrangement(X, axis=1)
            
            tmp = []            
            for i_channel in range(channels):
                for i_slice in range(0, slices, slices//r):
                    for i_column in range(0, columns, columns//r):
                        for i_row in range(0, rows, rows//r):
                            tmp.append(tf.slice(X, [0, i_row, i_column, i_slice, i_channel], [bsize, rows//r, columns//r, slices//r, 1]))
            X = tf.concat([x for x in tmp], axis=-1)
            return X
        
        # dense upsampling convoluiton (reference: https://github.com/tetrachrome/subpixel)
        def _subpixel(X, r):
            bsize, rows, columns, slices, channels = X.get_shape().as_list()
            X = tf.reshape(X, (bsize, rows, columns, slices, r, r, r, -1))
            X = tf.split(X, num_or_size_splits=rows, axis=1)  # row, [bsize, column, slices, r, r, r, 2]
            X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, column, slices, row*r, r, r, 2
            X = tf.split(X, num_or_size_splits=columns, axis=1)  # column, [bsize, slices, row*r, r, r, 2]
            X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, slices, row*r, column*r, r, 2
            X = tf.split(X, num_or_size_splits=slices, axis=1)  # slice, [bsize, row*r, column*r, r, 2]
            X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, slices, row*r, column*r, slices*r, 2
            return X
        
        ### conv_00
        params = self.layers[0]
        conv_layer = ConvolutionalLayer(n_output_chns=params['n_features'],
                                        kernel_size=params['kernel_size'],
                                        acti_func=self.acti_func,                                        
                                        with_bn=False,
                                        group_size=4,
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        name=params['name'])
        flow = conv_layer(images)
        layer_instances.append((conv_layer, flow))
        
        ### ddc_01
        params = self.layers[1]
        ddc_layer = LayerFromCallable(_reverse_subpixel, name=params['name'])
        flow = ddc_layer(flow, r=params['downsampling_rate'])
        layer_instances.append((ddc_layer, flow))
        
        ### lat_02: lateral_flow0
        params = self.layers[2]
        conv_layer = ConvolutionalLayer(n_output_chns=params['n_features'],
                                        kernel_size=params['kernel_size'],
                                        acti_func=self.acti_func,                                        
                                        with_bn=False,
                                        group_size=4,
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        name=params['name'])
        lateral_flow0 = conv_layer(flow)
        
        ### res_03 contains 3 resblocks, 1 resblock contains 2 convs(3x3x3)(dr=1,1). (dr: dilated rate)
        params = self.layers[3]
        for i in range(params['n_blocks']):
            res_block = HighResBlock(n_output_chns=params['n_features'],
                                     kernels=params['kernels'],
                                     dilated_rates=params['dilated_rates'],
                                     acti_func=self.acti_func,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     name='%s_%d' % (params['name'], i))
            flow = res_block(flow)
            layer_instances.append((res_block, flow))
            
        ### lateral_flow1
        lateral_flow1 = flow
        
        ### res_04 contains 3 resblocks, 1 resblock contains 2 convs(3x3x3)(dr=1,3). (dr: dilated rate)
        params = self.layers[4]
        for i in range(params['n_blocks']):
            res_block = HighResBlock(n_output_chns=params['n_features'],
                                     kernels=params['kernels'],
                                     dilated_rates=params['dilated_rates'],
                                     acti_func=self.acti_func,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     name='%s_%d' % (params['name'], i))
            flow = res_block(flow)
            layer_instances.append((res_block, flow))
            
        ### lateral_flow2
        lateral_flow2 = flow

        ### three branches (ptv, ctv, gtv)
        flow_dict = {'ptv': flow, 'ctv': flow, 'gtv': flow}
        
        ### res_05 contains 3 resblocks, 1 resblock contains 3 convs(3x3x3)(dr=1,3,5). (dr: dilated rate)
        params = self.layers[5]
        for flow_name in ['ptv', 'ctv', 'gtv']:
            for i in range(params['n_blocks']):
                res_block = HighResBlock(n_output_chns=params['n_features'],
                                         kernels=params['kernels'],
                                         dilated_rates=params['dilated_rates'],
                                         acti_func=self.acti_func,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         name='%s_%s_%d' % (flow_name, params['name'], i))
                flow_dict[flow_name] = res_block(flow_dict[flow_name])
                layer_instances.append((res_block, flow_dict[flow_name]))
        
        ### concatenate low-level feature maps (lateral_flow1, lateral_flow2, lateral_flow3) and high-level feature maps
        for flow_name in ['ptv', 'ctv', 'gtv']:
            flow_dict[flow_name] = tf.concat([lateral_flow0, lateral_flow1, lateral_flow2, flow_dict[flow_name]], axis=-1)
        
        ### conv_06 (fully convolutional layer)
        params = self.layers[6]
        for flow_name in ['ptv', 'ctv', 'gtv']:
            conv_layer = ConvolutionalLayer(n_output_chns=params['n_features'],
                                            kernel_size=params['kernel_size'],
                                            acti_func=self.acti_func,                                        
                                            with_bn=False,
                                            group_size=4,
                                            w_initializer=self.initializers['w'],
                                            w_regularizer=self.regularizers['w'],
                                            name='%s_%s' % (flow_name, params['name']))
            flow_dict[flow_name] = conv_layer(flow_dict[flow_name])
            layer_instances.append((conv_layer, flow_dict[flow_name]))
        
        ### conv_07 + duc_08 (output)
        params7 = self.layers[7]
        params8 = self.layers[8]
        for flow_name in ['ptv', 'ctv', 'gtv']:
            conv_layer = ConvolutionalLayer(n_output_chns=params7['n_features']*(params8['upsampling_rate']**3),
                                            kernel_size=params7['kernel_size'],
                                            acti_func=None,
                                            with_bn=False,
                                            group_size=4,
                                            w_initializer=self.initializers['w'],
                                            w_regularizer=self.regularizers['w'],
                                            name='%s_%s' % (flow_name, params7['name']))
            flow_dict[flow_name] = conv_layer(flow_dict[flow_name])
            layer_instances.append((conv_layer, flow_dict[flow_name]))
            duc_layer = LayerFromCallable(_subpixel, name=params8['name'])
            flow_dict[flow_name] = duc_layer(flow_dict[flow_name], r=params8['upsampling_rate'])
            layer_instances.append((duc_layer, flow_dict[flow_name]))
        
        flow = tf.concat([flow_dict['ptv'], flow_dict['ctv'], flow_dict['gtv']], axis=-1)
        
        # set training properties
        if is_training:
            self._print(layer_instances)
            return flow
        return flow_dict['gtv']
        
    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
            
class HighResBlock(TrainableLayer):
    def __init__(self,
                 n_output_chns,
                 kernels,
                 dilated_rates,
                 acti_func='prelu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):

        super(HighResBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        if type(kernels) is not list:
            raise ValueError('The parameter \'kernels\' must be a list containing each subkernel\'s kernel size.')
        if type(dilated_rates) is not list:
            raise ValueError('The parameter \'dilated_rates\' must be a list containing each subkernel\'s dilation rate.')
        assert (len(kernels) == len(dilated_rates)) # each subkernel only has one dilation rate
        
        self.kernels = kernels
        self.dilated_rates = dilated_rates
        self.acti_func = acti_func
        self.with_res = with_res
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor):
        output_tensor = input_tensor
        for (i, k), dr in zip(enumerate(self.kernels), self.dilated_rates):
            # create parameterised layers
            gn_op = GNLayer(group_size=4,
                            regularizer=self.regularizers['w'],
                            name='gn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=k,
                                dilation=dr,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = gn_op(output_tensor)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor
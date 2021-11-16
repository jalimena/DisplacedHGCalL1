from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import *

from tensorflow_model_optimization.python.core.quantization.keras import *
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as quantize_config_mod




class OutputOnlyConfig(quantize_config_mod.QuantizeConfig):

    def __init__(self, quantize_config):
      self.quantize_config = quantize_config

    def get_weights_and_quantizers(self, layer):
      return []

    def set_quantize_weights(self, layer, quantize_weights):
      pass

    def get_activations_and_quantizers(self, layer):
      return self.quantize_config.get_activations_and_quantizers(layer)

    def set_quantize_activations(self, layer, quantize_activations):
      return self.quantize_config.set_quantize_activations(
          layer, quantize_activations)

    def get_output_quantizers(self, layer):
      return self.quantize_config.get_output_quantizers(layer)

    def get_config(self):
      return {'quantize_config': self.quantize_config}

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation}
global_layers_list['QuantizeWrapper']=quantize_wrapper.QuantizeWrapper
global_layers_list['Default8BitQuantizeConfig']=default_8bit.default_8bit_quantize_registry.Default8BitQuantizeConfig
global_layers_list['QuantizeAwareActivation']=quantize_aware_activation.QuantizeAwareActivation
global_layers_list['OutputOnlyConfig']=OutputOnlyConfig
global_layers_list['Default8BitConvQuantizeConfig']=default_8bit.default_8bit_quantize_registry.Default8BitConvQuantizeConfig
global_layers_list['QConv2DBatchnorm']=QConv2DBatchnorm

from tensorflow.keras.layers import Layer
import tensorflow as tf

class Select8Layers(Layer):
    def __init__(self,**kwargs):
        super(Select8Layers, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape): # B x eta x phi x layers
        shape = input_shape
        shape[-1] = 8
        return shape 
        
    def call(self, input):
        l0 =  input[...,0:1]
        l2 =  input[...,2:3]
        l4 =  input[...,4:5]
        l6 =  input[...,6:7]
        l8 =  input[...,8:9]
        l10 = input[...,10:11]
        l12 = input[...,12:13]
        l13 = input[...,13:14]
        
        return tf.concat([l0,l2,l4,l6,l8,l10,l12,l13],axis=-1)

global_layers_list['Select8Layers']=Select8Layers


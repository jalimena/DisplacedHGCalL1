import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import time
from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout #etc
from Losses import binary_cross_entropy_with_extras


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks


#Apply the pruning to all Dense and conv2d layers
def apply_pruning(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  if isinstance(layer, tf.keras.layers.Conv2D):
      return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer


def my_model(Inputs,dropoutrate=0.01,momentum=0.95):
    model_path="../DisplacedHGCalL1/HGCalL1Images/Train/to_do/changeLayerSize/TrainOutput_add_block/KERAS_check_model_last.h5"
    orig_model = keras.models.load_model(model_path)
    model_for_pruning = tf.keras.models.clone_model(orig_model, clone_function=apply_pruning)
    x = Inputs[0]
    x = model_for_pruning(x)
    predictions=[x]
    return Model(inputs=Inputs, outputs=predictions)

train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model)


    train.compileModel(learningrate=0.0001,
                   loss='binary_crossentropy') #,binary_cross_entropy_with_extras)

print(train.keras_model.summary())

start = time.time()
train.change_learning_rate(0.0003)
model,history = train.trainModel(nepochs=10,
                                 batchsize=500,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_callbacks=pruning_callbacks.UpdatePruningStep())

                                                                                                

end = time.time()
print("time: ", end - start)


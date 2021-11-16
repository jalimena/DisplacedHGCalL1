import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
from DeepJetCore import TrainData
import matplotlib.pyplot as plt
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.training.gpuTools import DJCSetGPUs
import matplotlib

DJCSetGPUs("6")

#take argument from command line
model_path = str(sys.argv[1])
orig_model = keras.models.load_model(model_path)


#get input image and convert to numpy array
data_path = str(sys.argv[2])
td = TrainData()
td.readFromFile(data_path)
feature = np.array(td.copyFeatureListToNumpy(False))
truth = np.array(td.copyTruthListToNumpy(False))



#output after...
layers_models = ["pseudocolors", "dropout", "conv2d", "max_pooling2d", "conv2d_1", "max_pooling2d_1", "dropout_1", "conv2d_2", "max_pooling2d_2", "conv2d_3", "max_pooling2d_3", "dropout_2"]

batch_normalization0 = tf.keras.Model(orig_model.inputs, orig_model.layers[1].output)

dense0 = tf.keras.Model(orig_model.inputs, orig_model.layers[2].output)
dense1 = tf.keras.Model(orig_model.inputs, orig_model.layers[3].output)
pseudocolors = tf.keras.Model(orig_model.inputs, orig_model.layers[4].output)

dropout0 = tf.keras.Model(orig_model.inputs, orig_model.layers[5].output)

conv0 = tf.keras.Model(orig_model.inputs, orig_model.layers[6].output)
max_pooling0 = tf.keras.Model(orig_model.inputs, orig_model.layers[7].output)
batch_normalization1 = tf.keras.Model(orig_model.inputs, orig_model.layers[8].output)

conv1 = tf.keras.Model(orig_model.inputs, orig_model.layers[9].output)
max_pooling1 = tf.keras.Model(orig_model.inputs, orig_model.layers[10].output)
batch_normalization2 = tf.keras.Model(orig_model.inputs, orig_model.layers[11].output)

dropout1 = tf.keras.Model(orig_model.inputs, orig_model.layers[12].output)

conv2 = tf.keras.Model(orig_model.inputs, orig_model.layers[13].output)
max_pooling2 = tf.keras.Model(orig_model.inputs, orig_model.layers[14].output)
batch_normalization3 = tf.keras.Model(orig_model.inputs, orig_model.layers[15].output)

conv3 = tf.keras.Model(orig_model.inputs, orig_model.layers[16].output)
max_pooling3 = tf.keras.Model(orig_model.inputs, orig_model.layers[17].output)
batch_normalization4 = tf.keras.Model(orig_model.inputs, orig_model.layers[18].output)

dropout2 = tf.keras.Model(orig_model.inputs, orig_model.layers[19].output)

flatten = tf.keras.Model(orig_model.inputs, orig_model.layers[20].output)
dense2 = tf.keras.Model(orig_model.inputs, orig_model.layers[21].output)
dense3 = tf.keras.Model(orig_model.inputs, orig_model.layers[22].output)


truth_dwn = truth[0][0:1000].copy()

layer0 = []
layer1 = []
layer2 = []
layer3 = []
layer4 = []
layer5 = []
layer6 = []
layer7 = []
layer8 = []
layer9 = []
layer10 = []
layer11 = []
layer12 = []
layer13 = []
layer14 = []
layer15 = []
layer16 = []
layer17 = []
layer18 = []
layer19 = []
layer20 = []
layer21 = []



background = 0
for i in range(0, 1000):
    
    #Input image
    inputImage = feature[0][i] # 30 x 128 x 14

    if True:
        # Predict
        batch0_pred = batch_normalization0.predict(inputImage[None,:])

        dense0_pred = dense0.predict(inputImage[None,:])
        dense1_pred = dense1.predict(inputImage[None,:])
        pseudo_pred = pseudocolors.predict(inputImage[None,:])

        dropout0_pred = dropout0.predict(inputImage[None,:])

        conv0_pred = conv0.predict(inputImage[None,:])
        max_pooling0_pred = max_pooling0.predict(inputImage[None,:])
        batch1_pred = batch_normalization1.predict(inputImage[None,:])

        conv1_pred = conv1.predict(inputImage[None,:])
        max_pooling_pred1 = max_pooling1.predict(inputImage[None,:])
        batch2_pred = batch_normalization2.predict(inputImage[None,:])


        dropout1_pred = dropout1.predict(inputImage[None,:])

        conv2_pred = conv2.predict(inputImage[None,:])
        max_pooling_pred2 = max_pooling2.predict(inputImage[None,:])
        batch3_pred = batch_normalization3.predict(inputImage[None,:])

        conv3_pred = conv3.predict(inputImage[None,:])
        max_pooling_pred3 = max_pooling3.predict(inputImage[None,:])
        batch4_pred = batch_normalization4.predict(inputImage[None,:])

        dropout2_pred = dropout2.predict(inputImage[None,:])

        dense2_pred = dense2.predict(inputImage[None,:])
        dense3_pred = dense3.predict(inputImage[None,:])

    ###################################################################
        layer0.append(inputImage.copy())

        layer1.append(batch0_pred.copy())

        layer2.append(dense0_pred.copy())
        layer3.append(dense1_pred.copy())
        layer4.append(pseudo_pred.copy())

        layer5.append(dropout0_pred.copy())

        layer6.append(conv0_pred.copy())
        layer7.append(max_pooling0_pred.copy())
        layer8.append(batch1_pred.copy())

        layer9.append(conv1_pred.copy())
        layer10.append(max_pooling_pred1.copy())
        layer11.append(batch2_pred.copy())
        
        layer12.append(dropout1_pred.copy())
        
        layer13.append(conv2_pred.copy())
        layer14.append(max_pooling_pred2.copy())
        layer15.append(batch3_pred.copy())
        
        layer16.append(conv3_pred.copy())
        layer17.append(max_pooling_pred3.copy())
        layer18.append(batch4_pred.copy())


        layer19.append(dropout2_pred.copy())

        layer20.append(dense2_pred.copy())
        layer21.append(dense3_pred.copy())
    else:
        background += 1


print("saving features...")
np.savez_compressed('layers.npz', 
        layer0=np.asarray(layer0), 
        layer1=np.asarray(layer1), 
        layer2=np.asarray(layer2), 
        layer3=np.asarray(layer3), 
        layer4=np.asarray(layer4), 
        layer5=np.asarray(layer5),
        layer6=np.asarray(layer6),
        layer7=np.asarray(layer7),
        layer8=np.asarray(layer8),
        layer9=np.asarray(layer9),
        layer10=np.asarray(layer10),
        layer11=np.asarray(layer11),
        layer12=np.asarray(layer12),
        layer13=np.asarray(layer13),
        layer14=np.asarray(layer14),
        layer15=np.asarray(layer15),
        layer16=np.asarray(layer16),
        layer17=np.asarray(layer17),
        layer18=np.asarray(layer18),
        layer19=np.asarray(layer19),
        layer20=np.asarray(layer20),
        layer21=np.asarray(layer21)
        )

print("saving truths...")
np.savez_compressed('truths.npz', truths=truth_dwn)

#print("background events: ", background)
#print("signal events: ", 1000-background)
print("done")
































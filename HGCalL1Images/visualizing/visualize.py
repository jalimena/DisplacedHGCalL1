import numpy as np 
from matplotlib import pyplot as plt

with np.load('layers.npz') as features:
    image = features['layer0'] # 100 x 30 x 128 x 4
    pseudo = features['layer4'] # 100 x 1 x 30 x 128 x 4
    conv0 = features['layer6']
    conv1 = features['layer9']
    conv2 = features['layer13']
    conv3 = features['layer16']
    dense3 = features['layer21']

with np.load('truths.npz') as truths:
    labels = truths['truths']

print(image.shape, pseudo.shape, conv0.shape, labels.shape)



for n in range(0,1000):
    if labels[n][0] == 0 and dense3[n][0][0] > 0.8:
        print("gi")
        layers = [image, pseudo, conv0, conv1, conv2, conv3]
        #running = image[20,:,:,0]

        fig, axes  = plt.subplots(16, 7, figsize = (16, 9))
        axes[0,0].set_title('Input', pad=0, y=1.1)
        axes[0,1].set_title('pseudo colors', pad=0, y=1.1)
        axes[0,2].set_title('conv0', pad=0, y=1.1)
        axes[0,3].set_title('conv1', pad=0, y=1.1)
        axes[0,4].set_title('conv2', pad=0, y=1.25)
        axes[0,5].set_title('conv3', pad=0, y=1.1)
        axes[0,6].set_title('final dense', pad=0, y=1)
        #axes[0,7].set_title('final dense', pad=0, y=.8)
        

        for row in range(0, 16):
            for col in range(0, 7):
                axes[row, col].axis('off')
        
        axes[6,6].set_title(str(dense3[n][0][0]))
       # axes[0, 7].set_title('final dense', pad=0, y=.8, fontsize=10)
        for row in range(1, 15):
            axes[row, 0].imshow(image[n, :, :, row-1])
            #running = np.add(image[20,:,:,row], running)    
            
        for row in range(6,10):
            axes[row,1].imshow(pseudo[n, 0, :, :, row - 6])

        for row in range(0, 16):
            axes[row,2].imshow(conv0[n, 0, :, :, row])
            axes[row,3].imshow(conv1[n, 0, :, :, row])
            axes[row,4].imshow(conv2[n, 0, :, :, row])
            axes[row,5].imshow(conv3[n, 0, :, :, row])

        axes[7,6].imshow(dense3[n], vmin=0, vmax=1)
        print(dense3[n])
        print(labels[n])
        #plt.text(10, 10, "hi")
        plt.subplots_adjust(top=0.98,
            bottom=0.02,
            left=0.005,
            right=0.985,
            hspace=0.02,
            wspace=0.02)
        plt.show()


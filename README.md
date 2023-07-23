The code in this repository can be used to train two neural networks. 
The first neural network is a U-net, which allows semantic segmentation, in this case for a droplet.
Secondly, the mask of the droplet is used to compute the surface tension from it.

Both models can be trained by using the files Unet.ipynb and CNN_tens_drop.ipynb respectively.

Finally, in the file Predict.ipynb, the performance of the two sequential networks can be evualated.

For the complete explanation of the used pipeline and results of the project can be read in surfaceTensionFromDropletImage.pdf


The code for creating the droplet images can be found in the folder dropletGeneration.
To run this process one should open 'main.py' and run the following command:
    blender -b MultipleScripts_noDrop.blend  --python 'main.py'

Blender needs to be installed for this.


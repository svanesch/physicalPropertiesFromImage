import numpy as np
import bpy
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )

from gen_single_drop import genSingleDrop
from gen_drop_blender import Generate_Drop
from gen_drop_blender import Remove_Drop
from gen_drop_blender import Generate_Mask
from render import renderImages

# Define the parameters physical parameters
volume0=5
rneedle=0.5

sigma = np.array([40])

# Loop over different values for sigma
for i in range(0,len(sigma)):
        # Get rr and zz coordinates for droplet contour
        rr,zz = genSingleDrop(sigma[i],volume0,rneedle,output=0,savepath='.')

        # Create 3d droplet in blender
        Generate_Drop(rr,zz)

        # Create renders
        renderImages(sigma[i], volume0, rneedle, 'pathToStoreImages')

        # Generate the mask of the droplet
        Generate_Mask()
        renderImages(sigma[i], volume0, rneedle, 'pathToStoreImages')

        # Delete droplet in blender
        Remove_Drop()




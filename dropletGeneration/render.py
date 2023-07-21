import bpy
import numpy as np
import math

def renderImages(sigma, volume0, rneedle, output_dir, sun_strength = 6, cam_steps=5, camx_steps=3, sun_steps = 4, sun_rotate = 360, sun_angle=180, subject = bpy.context.object):
  import os
  sun = bpy.data.lights["Sun"]
  bpy.ops.object.select_all(action='DESELECT')
  cursor_loc = bpy.context.scene.cursor.location
  # Changes in strength of the sun
  for step in range(5,sun_strength):
    # Changes in sun position around Z axis
    for pos in range(sun_steps):
        bpy.data.objects['Sun'].select_set(True)  
        bpy.data.objects['Camera'].select_set(False)  
        bpy.ops.transform.rotate(value=(np.pi*(sun_rotate/sun_steps)/180), orient_axis = 'Z', center_override=cursor_loc) 
        sun.energy = step        
        # Changes in camera position around Y axis
        for cam in range(cam_steps):   
          bpy.data.objects['Sun'].select_set(False)  
          bpy.data.objects['Camera'].select_set(True)  
          bpy.ops.transform.rotate(value=(np.pi*(sun_rotate/cam_steps)/180), orient_axis = 'Y', center_override=cursor_loc) 
          # Changes in camera position around X axis
          for camx in range(camx_steps):   
            bpy.data.objects['Sun'].select_set(False)  
            bpy.data.objects['Camera'].select_set(True)  
            bpy.ops.transform.rotate(value=(np.pi*(sun_rotate/camx_steps)/180), orient_axis = 'X', center_override=cursor_loc) 
            # Saving renders with properties of that render
            bpy.context.scene.render.filepath = os.path.join(output_dir, ('drop' + '_s{}' + '_v{}' + '_r{}' + '_str{}' + '_pos{}' + '_cam{}'  + '_camx{}' + '.png').format(sigma, volume0, rneedle, step, pos, cam, camx))
            bpy.ops.render.render(write_still = True)
 

import numpy as np
import bpy

# Create 3d droplet in blender based on rr and zz
def Generate_Drop(rr,zz):
    cu = bpy.data.curves.new("poly", 'CURVE')
    cu.dimensions  = '3D'
    
    points = np.array([(rr[z],0,zz[z]) for z in range(len(zz))])

    s = cu.splines.new('BEZIER')
    s.bezier_points.add(len(points) - 1)
    s.bezier_points.foreach_set("co", points.flatten())

    for bp in s.bezier_points:
        bp.handle_left_type = bp.handle_right_type = 'AUTO'

    bpy.ops.curve.primitive_bezier_curve_add()
    ob = bpy.context.object
    ob.data = cu

    revolve = ob.modifiers.new("Screw", 'SCREW')
    revolve.steps = 128
    revolve.render_steps = 128
    
    bpy.data.objects['Needle'].hide_render = False
    bpy.context.scene.world = bpy.data.worlds['World']
    mat = bpy.data.materials.get("Droplet")
    if ob.data.materials:
        idx = ob.active_material_index
        ob.material_slots[idx].material = mat
    else:
        ob.data.materials.append(mat)

# Generate mask by changing the background world and droplet material
def Generate_Mask():
    bpy.data.objects['Needle'].hide_render = True 
    bpy.context.scene.world = bpy.data.worlds['World_Mask']
    ob = bpy.context.object
    if ob.data.materials:
        mat = bpy.data.materials.get("Droplet_Mask")
        idx = ob.active_material_index
        ob.material_slots[idx].material = mat
    else:
        # no slots
        ob.data.materials.append(mat)

# Remove the created droplet
def Remove_Drop():
    if bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['BezierCurve'].select_set(True)
    bpy.ops.object.delete()

import blenderproc as bproc
import bpy
import numpy as np

bproc.init()
objs = bproc.loader.load_blend('/home/david/BlenderProc/untitled.blend')

# Load Objects and Camera pose
dim = 2

for i in range(1):
	def sample_pose(obj: bproc.types.MeshObject):
	    obj.set_location(np.random.uniform([-dim,-dim,-dim], [dim,dim,dim]))
	    obj.set_rotation_euler(bproc.sampler.uniformSO3())

	bproc.object.sample_poses(objs, sample_pose_func=sample_pose)

	LocationCamera = bproc.sampler.shell(center=[0,0,0], radius_min=8, radius_max=8, elevation_min=-89, elevation_max=89)
		                                       
	#Compute rotation based on vector going from location towards poi
	rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - LocationCamera, inplane_rot=np.random.uniform(-0.7854, 0.7854))

	# Add homog cam pose based on location an rotation
	cam2world_matrix = bproc.math.build_transformation_mat(LocationCamera, rotation_matrix)

	bproc.camera.add_camera_pose(cam2world_matrix)

scene = bpy.data.scenes['Scene']
scene_collection = scene.collection

Objects = bpy.data.collections.new('Objects')
scene_collection.children.link(Objects)
LineArt = bpy.data.collections.new('LineArt')
scene_collection.children.link(LineArt)


mat1 = bpy.data.materials.new('Category1')
bpy.data.materials.create_gpencil_data(mat1)
mat1.grease_pencil.color = np.array([1,1,1,1])


for obj in objs:
    bpy.data.collections['Objects'].objects.link(bpy.data.objects[obj.get_name()])
    scene.collection.objects.unlink(bpy.data.objects[obj.get_name()])
    
    name = obj.get_name()+'_gpencil'
    gpencil = bpy.data.grease_pencils.new(name)
    gpencil_layer = gpencil.layers.new(name)
    gpencil_obj = bpy.data.objects.new(name,gpencil)  
    
    bpy.data.collections['LineArt'].objects.link(bpy.data.objects[name])
    
    LineArt = gpencil_obj.grease_pencil_modifiers.new(name,'GP_LINEART')
    
    LineArt.source_type = 'OBJECT'
    LineArt.source_object = bpy.data.objects[obj.get_name()]
    
    gpencil_obj.data.materials.append(bpy.data.materials['Category1'])
    LineArt.target_material = bpy.data.materials['Category1']
    
    LineArt.target_layer = gpencil.name
    
    LineArt.stroke_depth_offset = 0
    gpencil_layer.use_lights = False
    
scene.grease_pencil_settings.antialias_threshold = 0


scene.view_layers.new('edge')

image_render_layer = scene.node_tree.nodes['Render Layers']
image_render_layer.name = 'Image Render Layers'
edge_render_layer = scene.node_tree.nodes.new('CompositorNodeRLayers')
edge_render_layer.name = 'Edge Render Layers'
edge_render_layer.layer = 'edge'

scene.node_tree.nodes.new("CompositorNodeOutputFile")
output_file_edges = scene.node_tree.nodes['File Output']
scene.node_tree.links.new(edge_render_layer.outputs['Image'], output_file_edges.inputs['Image'])
output_file_edges.base_path = '/home/david/BlenderProc'
output_file_edges.file_slots['Image'].path = ''
output_file_edges.format.file_format = 'PNG'
output_file_edges.format.color_mode = 'BW'

# Already transformed to new file
scene.render.film_transparent = True

bpy.ops.object.lineart_bake_strokes_all()
scene.view_layers['edge'].layer_collection.children['Objects'].exclude = True
scene.view_layers['ViewLayer'].layer_collection.children['LineArt'].exclude = True
data = bproc.renderer.render()

import blenderproc as bproc
import os
import random
import numpy as np
import argparse
import bpy
from PIL import Image, ImageOps, ImageFile
import sys
import time

start = time.time()

#ImageFile.LOAD_TRUNCATED_IMAGES = True
np.set_printoptions(threshold=sys.maxsize)

#TODO TODAY: SCREW DETECTOR Anschauen

# TODO
# Range for Camera Options and remove ratio
# What is alpha channel form texture ?
# Coco Annotation writer: Check Results
# Check all rooms and set dimensions in which objects can be placed or check new implementation wir ceiling
# Freestyle Problematic
# iterate through different rooms

#base_path = '/mnt/extern_hd/BPRandomRoom'
base_path = '/home/david/BlenderProc/SceneNet'
parser = argparse.ArgumentParser()
parser.add_argument('scene_net_obj_path',nargs='?', default= os.path.join(base_path,'Scenet','Bedroom','xih07063_labels.obj'), help="Path to the used scene net `.obj` file, download via scripts/download_scenenet.py")
parser.add_argument('scene_texture_path',nargs='?', default= os.path.join(base_path,'texture_library'), help="Path to the downloaded texture files, you can find them at http://tinyurl.com/zpc9ppb")
parser.add_argument('cc_material_path', nargs='?', default=os.path.join(base_path,'CCTexture'), help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument('objects', nargs = '?', default=os.path.join(base_path,'wood_element_data','wood_element.blend'),help = "Path to the objects of the tracked objects")
parser.add_argument('output_dir', nargs='?', default=os.path.join(base_path,'output'), help="Path to where the final files, will be saved")
args = parser.parse_args()

# Create Path for Output
class_annotation_path = os.path.join(args.output_dir,'class_annotation')
instance_annotation_path = os.path.join(args.output_dir,'instance_annotation')
images_path = os.path.join(args.output_dir,'images')

if not os.path.exists(class_annotation_path):
   os.makedirs(class_annotation_path)
if not os.path.exists(instance_annotation_path):
   os.makedirs(instance_annotation_path)
if not os.path.exists(images_path):
   os.makedirs(images_path)

# rooms:




bproc.init()

bpy.ops.scene.freestyle_linestyle_new()


# Some Options
#####################

num_camera = 1
num_poses = 1

# rendering options used as in bproc
scene = bpy.data.scenes['Scene']
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'
bproc.renderer.set_noise_threshold(0.05)
# TODO
#bproc.renderer.set_max_amount_of_samples(1024)
bproc.renderer.set_max_amount_of_samples(10)
bproc.renderer.set_denoiser("INTEL")

# Setting for imagesize and Resolution: Phone
image_width = 720
image_height = 1280
fy = 960.674
fx = 960.367
cy = 652.814
cx = 361.234
fy_range = [960, 961]
fx_range = [960, 961]
cy_range = [650, 654]
cx_range = [360, 362]

# Setting for imagesize and Resolution: Ground Truth: Real: 512,256
ratio = 1/2
####################

# Load the scenenet room and label its objects with category ids based on the nyu mapping
label_mapping = bproc.utility.LabelIdMapping.from_csv(bproc.utility.resolve_resource(os.path.join('id_mappings', 'nyu_idset.csv')))
objs = bproc.loader.load_scenenet(args.scene_net_obj_path, args.scene_texture_path, label_mapping)

#Load Object to get 3D pose
objs_edge = bproc.loader.load_blend(args.objects)
sheet = bproc.filter.by_attr(objs_edge,"name","Wood_Sheet.*", regex=True)
bar = bproc.filter.by_attr(objs_edge,"name","Wood_Bar.*", regex=True)
frame = bproc.filter.by_attr(objs_edge,"name","Wood_Side.*", regex=True)
base = bproc.filter.by_attr(objs_edge,"name","Wood_Block.*", regex=True)


# set Id and objects to rigid bodies
for obj in base:
    obj.set_cp("category_id", 1)
for obj in bar:
    obj.set_cp("category_id", 2)
for obj in frame:
    obj.set_cp("category_id", 2)
for obj in sheet:
    obj.set_cp("category_id", 3)

for obj in objs_edge:
    obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", collision_margin=0.0025)



SecurityFloor = bproc.loader.load_obj(os.path.join(base_path,"floor.obj"))
for obj in SecurityFloor:
    obj.set_location([0, 0, -0.05])
    obj.enable_rigidbody(active=False, collision_shape="CONVEX_HULL")

i = 1
for obj in objs_edge:
    Collection = bpy.data.collections.new("Collection {}".format(i))
    Collection.objects.link(bpy.data.objects[obj.get_name()])
    i += 1

# Freestyle options
scene = bpy.data.scenes['Scene']
scene.render.use_freestyle = True
freestyle_settings = scene.view_layers['ViewLayer'].freestyle_settings
freestyle_settings.as_render_pass = True
freestyle_settings.crease_angle = 140/180*np.pi
freestyle_settings.use_culling = True

# Remove LineSet and create new LineSet with logical Names
freestyle_settings.linesets.remove(freestyle_settings.linesets['LineSet'])
i = 1
for collection in bpy.data.collections:
    if i <= len(objs_edge):
        lineset = freestyle_settings.linesets.new('LineSet {}'.format(i))
        lineset.select_silhouette = False
        lineset.select_crease = True
        lineset.select_border = True
        lineset.exclude_border = True
        lineset.edge_type_combination = 'AND'

        # lineset.linestyle.color = (10*i,10*i,10*i)
        lineset.linestyle.color = (1, 1, 1)
        lineset.linestyle.thickness = 1.0
        lineset.linestyle.thickness_position = 'INSIDE'

        lineset.select_by_collection = True
        lineset.collection = bpy.data.collections["Collection {}".format(i)]
        i += 1

# Save freestyle image
nodes = scene.node_tree.nodes
render_layers = nodes['Render Layers']
output_file_freestyle = nodes.new("CompositorNodeOutputFile")
output_file_freestyle.base_path = "./freestyle/"
scene.node_tree.links.new(render_layers.outputs['Freestyle'],output_file_freestyle.inputs['Image'])
output_file_freestyle.file_slots['Image'].path = 'Freestyle'
output_file_freestyle.format.file_format = 'PNG'
output_file_freestyle.format.color_mode = 'BW'



# Load all recommended cc materials, however don't load their textures yet
cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, preload=True)

# Go through all objects
for obj in objs:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # In 40% of all cases
        if np.random.uniform(0, 1) <= 0.1:
            # Replace the material with a random one from cc materials
            obj.set_material(i, random.choice(cc_materials))

        id = obj.get_cp("category_id")
        id_chair = label_mapping.id_from_label("chair")
        id_table = label_mapping.id_from_label("table")
        id_sofa = label_mapping.id_from_label("sofa")
        id_bed = label_mapping.id_from_label("bed")
        id_desk = label_mapping.id_from_label("desk")
        id_other_struct = label_mapping.id_from_label("otherstructure")
        id_other_furniture = label_mapping.id_from_label("otherfurniture")
        id_check = id == id_chair or id == id_table or id == id_sofa or id == id_bed or \
                id == id_desk or id_other_struct or id_other_furniture

        if id_check == id_chair or id == id_table or id == id_sofa or id == id_bed or id == id_desk:
            obj.enable_rigidbody(active=False)

# Now load all textures of the materials that were assigned to at least one object
bproc.loader.load_ccmaterials(args.cc_material_path, fill_used_empty_materials=True)

# In some scenes floors, walls and ceilings are one object that needs to be split first
# Collect all walls
walls = bproc.filter.by_cp(objs, "category_id", label_mapping.id_from_label("wall"))
# Extract floors from the objects
new_floors = bproc.object.extract_floor(walls, new_name_for_object="floor", should_skip_if_object_is_already_there=True)
# Set category id of all new floors
for floor in new_floors:
    floor.set_cp("category_id", label_mapping.id_from_label("floor"))
# Add new floors to our total set of objects
objs += new_floors

# Extract ceilings from the objects
new_ceilings = bproc.object.extract_floor(walls, new_name_for_object="ceiling", up_vector_upwards=False, should_skip_if_object_is_already_there=True)
# Set category id of all new ceiling
for ceiling in new_ceilings:
    ceiling.set_cp("category_id", label_mapping.id_from_label("ceiling"))
# Add new ceilings to our total set of objects
objs += new_ceilings


# Make all lamp objects emit light
lamps = bproc.filter.by_attr(objs, "name", ".*[l|L]amp.*", regex=True)
bproc.lighting.light_surface(lamps, emission_strength=15)
# Also let all ceiling objects emit a bit of light, so the whole room gets more bright
ceilings = bproc.filter.by_attr(objs, "name", ".*[c|C]eiling.*", regex=True)
bproc.lighting.light_surface(ceilings, emission_strength=2, emission_color=[1,1,1,1])

obj_all = objs+objs_edge
# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects(obj_all)

# Find all floors in the scene, so we can sample locations above them
floors = bproc.filter.by_cp(objs, "category_id", label_mapping.id_from_label("floor"))

# Reset all category id's:
# TODO REMOVE Comments
for obj in objs:
    obj.set_cp("category_id", 0)


# used to save images
img = 0


poses = 0
tries = 0

# Clear all key frames from the previous run
bproc.utility.reset_keyframes()

while tries < 10000 and poses < num_poses:

    bproc.utility.reset_keyframes()

    tries += 1
    # Sample point above the floor to let them fall down: 1 to 1.5 such that cube to sample objects completely inside of the room
    # Probably not even important
    min_height = 1
    max_height = 1.5
    #location = np.random([])
    location = bproc.sampler.upper_region(ceilings, min_height=0.5, max_height=1)
    location = location - [0, 0, 2]
    location_floor = [location[0], location[1], 0]

    # Check that there is no object at the floor where the objects fall down:
    #_, _, _, _, hit_object, _ = bproc.object.scene_ray_cast([location[0], location[1], 0.2], [0, 0, -1])
    #if hit_object not in floors:
    #    continue

    hit1, _, _, _, _, _ = bproc.object.scene_ray_cast(location, [1, 0, 0], 0.4)
    hit2, _, _, _, _, _ = bproc.object.scene_ray_cast(location, [-1, 0, 0], 0.4)
    hit3, _, _, _, _, _ = bproc.object.scene_ray_cast(location, [0, 1, 0], 0.4)
    hit4, _, _, _, _, _ = bproc.object.scene_ray_cast(location, [0, -1, 0], 0.4)

    if hit1 or hit2 or hit3 or hit4:
        continue

    # Define a function that samples 6-DoF poses
    cube_dim = np.random.uniform(0.2, 1)
    loc1 = location + [cube_dim, cube_dim, 0.8]
    loc2 = location - [cube_dim, cube_dim, 0.8]
    def sample_pose(obj: bproc.types.MeshObject):
        obj.set_location(np.random.uniform(loc1, loc2))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    # Sample the poses of all screw objects.
    bproc.object.sample_poses(objs_edge, sample_pose_func=sample_pose)

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)

    cam = 1
    while cam <= num_camera:

        # Sample location
        LocationCamera = bproc.sampler.shell(center=location_floor,
                                       radius_min=0.3,
                                       radius_max=1.5,
                                       elevation_min=25,
                                       elevation_max=89)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(location_floor - LocationCamera, inplane_rot=np.random.uniform(-0.7854, 0.7854))

        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(LocationCamera, rotation_matrix)

        if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.1}, bvh_tree):
            continue

        # TODO consider adding check that Camera is not outside of the wall

        bproc.camera.add_camera_pose(cam2world_matrix)
        cam += 1

    poses += 1

    # render the whole pipeline
    fx = np.random.uniform(fx_range[0], fx_range[1])
    fy = np.random.uniform(fy_range[0], fy_range[1])
    cx = np.random.uniform(cx_range[0], cx_range[1])
    cy = np.random.uniform(cy_range[0], cy_range[1])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)
    data = bproc.renderer.render()
    bproc.camera.set_intrinsics_from_K_matrix(K * ratio, int(image_width * ratio), int(image_height * ratio))
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    # Use Freestyle Image information together with segmentation to get semantic edges ground truth
    class_segmaps = data["class_segmaps"]
    instance_segmaps = data["instance_segmaps"]
    colors = data["colors"]
    attribute_maps = data["instance_attribute_maps"]
    num_frames = len(colors)

    # Reassign instance values:
    sheet = bproc.filter.by_attr(objs_edge, "name", "Wood_Sheet.*", regex=True)
    bar = bproc.filter.by_attr(objs_edge, "name", "Wood_Bar.*", regex=True)
    frame = bproc.filter.by_attr(objs_edge, "name", "Wood_Side.*", regex=True)
    base = bproc.filter.by_attr(objs_edge, "name", "Wood_Block.*", regex=True)
    new_attribute_maps = []
    mapping = []
    names = ["Wood_Block_Base", "Wood_Bar_Long.001", "Wood_Bar_Long.002", "Wood_Bar_Short.001", "Wood_Bar_Short.002"
             , "Wood_Side_Long.001", "Wood_Side_Long.002", "Wood_Side_Short.001", "Wood_Side_Short.002"
             , "Wood_Sheet_Large.001", "Wood_Sheet_Small.001", "Wood_Sheet_Large.001"]


    if len(colors) == len(class_segmaps):

        for i in range(num_camera):
            new_attribute_maps.append([])
            mapping.append([])
            for dict in attribute_maps[i]:
                for idx in range(len(names)):
                    if dict["name"] == names[idx]:
                        new_attribute_maps[i].append({"idx": idx, "category_id": dict["category_id"], "name": dict["name"]})
                        mapping[i].append({"old_idx": dict["idx"], "new_idx": idx})

        # Save in CaseNet format
        for i in range(num_camera):

            freestyle_image = Image.open('./freestyle/Freestyle{:04d}.png'.format(i))

            freestyle_image = np.array(freestyle_image, dtype=np.uint8)
            freestyle_image = np.where(freestyle_image <= 10, 0, 1)

            class_segmaps[i] = class_segmaps[i] * freestyle_image
            class_segmaps[i] = class_segmaps[i].astype(np.uint8)
            class_segmaps_image = Image.fromarray(class_segmaps[i])
            class_segmaps_image.save(class_annotation_path + '/{:04d}.png'.format(img))

            instance_segmaps[i] = np.array(instance_segmaps[i], dtype=np.uint8) * freestyle_image
            for maps in mapping[i]:
                instance_segmaps[i] = np.where((instance_segmaps[i] <= maps["old_idx"] + 0.5) & (instance_segmaps[i] >= maps["old_idx"] - 0.5), maps["new_idx"], instance_segmaps[i])
            instance_segmaps[i] = instance_segmaps[i].astype(np.uint8)
            instance_segmaps_image = Image.fromarray(instance_segmaps[i])
            instance_segmaps_image.save(instance_annotation_path + '/{:04d}.png'.format(img))

            color_image = Image.fromarray(np.array(colors[i], dtype=np.uint8))
            color_image.save(images_path + '/{:04d}.png'.format(img))

            img += 1


        data["class_segmaps"] = class_segmaps
        data["instance_segmaps"] = instance_segmaps
        data["instance_attribute_maps"] = new_attribute_maps

        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

        bproc.writer.write_coco_annotations(
            os.path.join(args.output_dir, 'coco_data'),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            append_to_existing_output=True,
            mask_encoding_format="rle",
            color_file_format="PNG")

    else:
        print("error occured during image segmentation in BlenderProc")


end = time.time()
with open('time.txt', 'w') as f:
    f.write('Have Computed {} images and used therefore {} '.format(poses*num_camera,end-start))


import blenderproc as bproc
import os
import shutil
import random
import numpy as np
import argparse
import bpy
import sys
import time
sys.path.append(os.getcwd())
import Helper

start = time.time()


# TODO
# Range for Camera Options and remove ratio
# Coco Annotation writer: Check Results
# iterate through different rooms

# Important parameter:
append_to_existing = False
num_camera = 4
num_poses_Bedroom = 1
num_poses_Living_Room = 5
num_poses_Office = 1

#base_path = '/mnt/extern_hd/BPRandomRoom'
base_path = '/home/david/BlenderProc/SceneNet'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('scene_net_path',nargs='?', default= os.path.join(base_path,'Scenet'), help="Path to the used scene net `.obj` file, download via scripts/download_scenenet.py")
parser.add_argument('scene_texture_path',nargs='?', default= os.path.join(base_path,'texture_library'), help="Path to the downloaded texture files, you can find them at http://tinyurl.com/zpc9ppb")
parser.add_argument('cc_material_path', nargs='?', default=os.path.join(base_path,'CCTexture'), help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument('objects', nargs = '?', default=os.path.join(base_path,'objects','wood_element.blend'),help = "Path to the objects of the tracked objects")
parser.add_argument('output_dir', nargs='?', default=os.path.join(base_path,'output'), help="Path to where the final files, will be saved")
args = parser.parse_args()

# Create Path for Output
paths_full = {"hdf5": os.path.join(args.output_dir, 'full', 'hdf5'),
              "coco": os.path.join(args.output_dir, 'full', 'coco'),
              "images": os.path.join(args.output_dir, 'full', 'images'),
              "class_annotation": os.path.join(args.output_dir, 'full', 'class_annotation'),
              "instance_annotation": os.path.join(args.output_dir, 'full', 'instance_annotation'),
              "freestyle": os.path.join(args.output_dir, 'full', 'freestyle')}

paths_half = {"hdf5": os.path.join(args.output_dir, 'half', 'hdf5'),
              "coco": os.path.join(args.output_dir, 'half', 'coco'),
              "images": os.path.join(args.output_dir, 'half', 'images'),
              "class_annotation": os.path.join(args.output_dir, 'half', 'class_annotation'),
              "instance_annotation": os.path.join(args.output_dir, 'half', 'instance_annotation'),
              "freestyle": os.path.join(args.output_dir, 'half', 'freestyle')}

for path in paths_half.values():
    if not os.path.exists(path):
        os.makedirs(path)

for path in paths_full.values():
    if not os.path.exists(path):
        os.makedirs(path)

# get current img_idx if append to existing = true
if append_to_existing:
    img_idx = Helper.get_idx(paths_full)
else:
    Helper.clean_output(paths_full)
    Helper.clean_output(paths_half)
    img_idx = 0

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

# get dictionary with possible rooms
rooms = {'Bedroom': [], 'Living-room': [], 'Office': []}
#rooms = {'Office': []}
for room_category in rooms.keys():
    for file in os.listdir(os.path.join(args.scene_net_path, room_category)):
        if file.endswith(".obj"):
            rooms[room_category].append(file)

# rooms:
for room_category, room_list in rooms.items():
    num_poses = num_poses_Office * (room_category == 'Office') + \
                num_poses_Living_Room * (room_category == 'Living-room') + \
                num_poses_Bedroom * (room_category == 'Bedroom')

    for room in room_list:
        # for poopooo in range(1):
        # room = room_list[0]

        # TODO consider only reloading some components
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bproc.init()

        bpy.ops.scene.freestyle_linestyle_new()

        # rendering options used as in bproc
        scene = bpy.data.scenes['Scene']
        scene.render.resolution_percentage = 100
        scene.render.engine = 'CYCLES'
        bproc.renderer.set_noise_threshold(0.05)
        # TODO
        # bproc.renderer.set_max_amount_of_samples(1024)
        bproc.renderer.set_max_amount_of_samples(10)
        bproc.renderer.set_denoiser("INTEL")

        # Load the scenenet room and label its objects with category ids based on the nyu mapping
        label_mapping = bproc.utility.LabelIdMapping.from_csv(bproc.utility.resolve_resource(os.path.join('id_mappings', 'nyu_idset.csv')))
        objs = bproc.loader.load_scenenet(os.path.join(args.scene_net_path, room_category, room), args.scene_texture_path, label_mapping)

        # Load Object to get 3D pose
        objs_edge = bproc.loader.load_blend(args.objects)
        sheet = bproc.filter.by_attr(objs_edge,"name","Wood_Sheet.*", regex=True)
        bar = bproc.filter.by_attr(objs_edge,"name","Wood_Bar.*", regex=True)
        frame = bproc.filter.by_attr(objs_edge,"name","Wood_Side.*", regex=True)
        base = bproc.filter.by_attr(objs_edge,"name","Wood_Block.*", regex=True)

        # randomize material
        Helper.material_randomizer()

        # set Id and objects to rigid bodies
        for obj in base:
            obj.set_cp("category_id", 1)
        for obj in bar:
            obj.set_cp("category_id", 2)
        for obj in frame:
            obj.set_cp("category_id", 2)
        for obj in sheet:
            obj.set_cp("category_id", 3)

        names = []
        for obj in objs_edge:
            obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", collision_margin=0.0025)
            names.append(obj.get_name())
            Collection = bpy.data.collections.new(obj.get_name())
            Collection.objects.link(bpy.data.objects[obj.get_name()])

        additional_floor = bproc.loader.load_obj(os.path.join(base_path, 'objects', "floor.obj"))
        for obj in additional_floor:
            obj.set_location([0, 0, -0.05])
            obj.enable_rigidbody(active=False, collision_shape="CONVEX_HULL")

        # Freestyle options
        scene = bpy.data.scenes['Scene']
        scene.render.use_freestyle = True
        freestyle_settings = scene.view_layers['ViewLayer'].freestyle_settings
        freestyle_settings.as_render_pass = True
        freestyle_settings.crease_angle = 140/180*np.pi
        freestyle_settings.use_culling = True

        # Remove LineSet and create new LineSet with logical Names
        print(freestyle_settings.linesets.keys())
        freestyle_settings.linesets.remove(freestyle_settings.linesets['LineSet'])

        for collection in bpy.data.collections:
            if collection.name != 'Wood_Block_Base' and collection.name != 'RigidBodyWorld':
                lineset = freestyle_settings.linesets.new(collection.name)
                lineset.edge_type_combination = 'AND'
                lineset.select_silhouette = False
                lineset.select_crease = True
                lineset.select_border = True
                lineset.exclude_border = True
                lineset.select_material_boundary = True
                lineset.exclude_material_boundary = True

                lineset.linestyle.color = (1, 1, 1)
                lineset.linestyle.thickness = 1.0
                lineset.linestyle.thickness_position = 'INSIDE'

                lineset.select_by_collection = True
                lineset.collection = bpy.data.collections[collection.name]
            elif collection.name == 'Wood_Block_Base':
                lineset = freestyle_settings.linesets.new(collection.name)
                lineset.edge_type_combination = 'AND'
                lineset.select_edge_mark = True

                lineset.linestyle.color = (1, 1, 1)
                lineset.linestyle.thickness = 1.0
                lineset.linestyle.thickness_position = 'INSIDE'

                lineset.select_by_collection = True
                lineset.collection = bpy.data.collections[collection.name]

        # Save freestyle image
        nodes = scene.node_tree.nodes
        render_layers = nodes['Render Layers']
        nodes.new("CompositorNodeOutputFile")
        output_file_freestyle = scene.node_tree.nodes['File Output']
        scene.node_tree.links.new(render_layers.outputs['Freestyle'], output_file_freestyle.inputs['Image'])
        output_file_freestyle.base_path = paths_full['freestyle']
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
        for obj in objs:
            obj.set_cp("category_id", 0)

        # used to save images
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

            hit = np.zeros(8)
            for i in range(hit.shape[0]):
                hit[i], _, _, _, _, _ = bproc.object.scene_ray_cast(location, [np.cos(i*2*np.pi/hit.shape[0]), np.sin(i*2*np.pi/hit.shape[0]), 0], 0.7)

            if hit.any():
                continue

            # Define a function that samples 6-DoF poses
            cube_dim = np.random.uniform(0.2, 0.7)
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
            bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_full['freestyle']
            data = bproc.renderer.render()
            seg_data_full = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])
            bproc.camera.set_intrinsics_from_K_matrix(K * ratio, int(image_width * ratio), int(image_height * ratio))
            bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_half['freestyle']
            seg_data_half = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])

            _ = Helper.save_data(data, seg_data_half, paths_half, names, img_idx)
            img_idx = Helper.save_data(data, seg_data_full, paths_full, names, img_idx)


end = time.time()
with open('time.txt', 'w') as f:
    f.write('Have Computed {} images and used therefore {} '.format(poses*num_camera,end-start))


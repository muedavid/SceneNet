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

np.set_printoptions(threshold=sys.maxsize)

# Important parameter:
append_to_existing = True
segmenter = False
testing_mode = False
num_camera = 2
num_poses_Bedroom = 1
num_poses_Living_Room = 4
num_poses_Office = 1
hit_radius = 1
#camera_radius = [0.2, 1.2]
camera_radius = [0.3, 1.2]
sample_cube_dims = [0.2, 0.8, 1.5]
samples = 512
num_training = 9
training = True
testing = False


if testing_mode:
    samples = 10
    num_poses_Living_Room = 1
    num_camera = 1

base_path = '/home/david/BlenderProc/SceneNet'
if not os.path.isdir(base_path):
    base_path = '/mnt/extern_hd/SceneNet'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('scene_net_path',nargs='?', default= os.path.join(base_path,'Scenet'), help="Path to the used scene net `.obj` file, download via scripts/download_scenenet.py")
parser.add_argument('scene_texture_path',nargs='?', default= os.path.join(base_path,'texture_library'), help="Path to the downloaded texture files, you can find them at http://tinyurl.com/zpc9ppb")
parser.add_argument('cc_material_path', nargs='?', default=os.path.join(base_path,'CCTexture'), help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument('objects', nargs = '?', default=os.path.join(base_path,'objects','wood_element.blend'),help = "Path to the objects of the tracked objects")
if training:
    parser.add_argument('output_dir', nargs='?', default=os.path.join(base_path,'outputTrain'), help="Path to where the final files, will be saved")
elif testing:
    parser.add_argument('output_dir', nargs='?', default=os.path.join(base_path, 'outputTest'),help="Path to where the final files, will be saved")
else:
    raise ValueError('training and testing are set to False')
args = parser.parse_args()

# Create Path for Output
paths_full = {"hdf5": os.path.join(args.output_dir, 'full', 'hdf5'),
              "coco": os.path.join(args.output_dir, 'full', 'coco'),
              "images": os.path.join(args.output_dir, 'full', 'images'),
              "class_annotation": os.path.join(args.output_dir, 'full', 'class_annotation'),
              "instance_annotation": os.path.join(args.output_dir, 'full', 'instance_annotation'),
              "line_art_class": os.path.join(args.output_dir, 'full', 'line_art', 'class'),
              "line_art_instance": os.path.join(args.output_dir, 'full', 'line_art', 'instance')}

paths_half = {"hdf5": os.path.join(args.output_dir, 'half', 'hdf5'),
              "coco": os.path.join(args.output_dir, 'half', 'coco'),
              "images": os.path.join(args.output_dir, 'half', 'images'),
              "class_annotation": os.path.join(args.output_dir, 'half', 'class_annotation'),
              "instance_annotation": os.path.join(args.output_dir, 'half', 'instance_annotation'),
              "line_art_class": os.path.join(args.output_dir, 'half', 'line_art', 'class'),
              "line_art_instance": os.path.join(args.output_dir, 'half', 'line_art', 'instance')}

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

# Setting for imagesize and Resolution: Tablet
image_width = 720
image_height = 1280
fy = 951.142
fx = 954.596
cy = 647.781
cx = 362.162
fy_range = [fy-1, fy+1]
fx_range = [fx-1, fx+1]
cy_range = [cy-1, cy+1]
cx_range = [cx-1, cx+1]

ratio = 1/2

# get dictionary with possible rooms
rooms = {'Bedroom': [], 'Living-room': [], 'Office': []}
if testing_mode:
    rooms = {'Bedroom': []}
    room_test = 0
    #rooms = {'Living-room': []}

# set steps of 5
color_mapping = {'1': 39, '2': 56, '3': 69, '4': 80, '5': 88, '6': 96,
                 '7': 104, '8': 111, '9': 117, '10': 123, '11': 129, '12': 133}

for room_category in rooms.keys():
    for file in os.listdir(os.path.join(args.scene_net_path, room_category)):
        if file.endswith(".obj"):
            rooms[room_category].append(file)

bproc.init()
bpy.ops.object.light_add(type="POINT", radius=1, align="WORLD", location=(0, 0, 8), scale=(1, 1, 1))
bpy.data.lights["Point"].energy = np.random.uniform(800, 1600)

bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.engine = 'CYCLES'
bproc.renderer.set_noise_threshold(0.05)
bpy.data.scenes['Scene'].render.film_transparent = True
# bproc.renderer.set_max_amount_of_samples(1024)
# TODO
if '/mnt/' in base_path:
    bpy.data.scenes['Scene'].cycles.device = 'CPU'
else:
    bpy.data.scenes['Scene'].cycles.device = 'GPU'
bproc.renderer.set_denoiser("INTEL")

scene_collection = bpy.data.scenes['Scene'].collection

# Define New View Layer for Implementation
Objects = bpy.data.collections.new('Objects')
scene_collection.children.link(Objects)
LineArt = bpy.data.collections.new('LineArt')
scene_collection.children.link(LineArt)

# Options for grease pencil
bpy.data.scenes['Scene'].grease_pencil_settings.antialias_threshold = 0

# Add View Layer for Rendering
bpy.data.scenes['Scene'].view_layers.new('edge')
edge_render_layer = bpy.data.scenes['Scene'].node_tree.nodes.new('CompositorNodeRLayers')
edge_render_layer.name = 'Edge Render Layers'
edge_render_layer.layer = 'edge'
image_render_layer = bpy.data.scenes['Scene'].node_tree.nodes['Render Layers']
image_render_layer.name = 'Image Render Layers'

# Define Output for rendered edge Images
bpy.data.scenes['Scene'].node_tree.nodes.new("CompositorNodeOutputFile")
output_file_edges = bpy.data.scenes['Scene'].node_tree.nodes['File Output']
bpy.data.scenes['Scene'].node_tree.links.new(edge_render_layer.outputs['Image'], output_file_edges.inputs['Image'])
output_file_edges.file_slots['Image'].path = ''
output_file_edges.format.color_mode = 'BW'

# Load Object to get 3D pose
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

names = []
for obj in objs_edge:
    obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", collision_margin=0.002)
    names.append(obj.get_name())

    # Set all Objects in a Collection: used for render layer
    bpy.data.collections['Objects'].objects.link(bpy.data.objects[obj.get_name()])
    bpy.data.scenes['Scene'].collection.objects.unlink(bpy.data.objects[obj.get_name()])

    # Create GPencil Object and LineArt Modifier
    name = obj.get_name() + '_gpencil'
    gpencil = bpy.data.grease_pencils.new(name)
    gpencil_layer = gpencil.layers.new(name)
    gpencil_obj = bpy.data.objects.new(name, gpencil)
    LineArt = gpencil_obj.grease_pencil_modifiers.new(name, 'GP_LINEART')

    # Set GPencil Object in a Collection: used for render layer
    bpy.data.collections['LineArt'].objects.link(bpy.data.objects[name])

    # Define Inputs into LineArt Modifier
    LineArt.source_type = 'OBJECT'
    LineArt.source_object = bpy.data.objects[obj.get_name()]

    # Define Material for gpencil for each object 1
    mat = bpy.data.materials.new(obj.get_name())
    bpy.data.materials.create_gpencil_data(mat)

    gpencil_obj.data.materials.append(mat)
    LineArt.target_material = mat

    LineArt.target_layer = gpencil.name

    # Options for LineArt
    LineArt.stroke_depth_offset = 0
    LineArt.thickness = 1
    LineArt.use_intersection = False
    LineArt.use_loose = False
    LineArt.use_material = False
    LineArt.use_contour = False
    LineArt.use_edge_overlap = True
    LineArt.use_crease = False
    gpencil_layer.use_lights = False
    gpencil.stroke_thickness_space = 'SCREENSPACE'
    gpencil.stroke_depth_order = '3D'

# get material color and pos
material_color, material_pos = Helper.get_material_color()

additional_floor = bproc.loader.load_obj(os.path.join(base_path, 'objects', "floor.obj"))
for obj in additional_floor:
    obj.set_location([0, 0, -0.0495])
    obj.enable_rigidbody(active=False, collision_shape="CONVEX_HULL", collision_margin=0.005)
    bpy.data.collections['Objects'].objects.link(bpy.data.objects[obj.get_name()])
    bpy.data.scenes['Scene'].collection.objects.unlink(bpy.data.objects[obj.get_name()])

# rooms:
for room_category, room_list in rooms.items():
    num_poses = num_poses_Office * (room_category == 'Office') + \
                num_poses_Living_Room * (room_category == 'Living-room') + \
                num_poses_Bedroom * (room_category == 'Bedroom')

    if testing_mode:
        room_list = room_list[room_test:room_test+1]
    elif training:
        room_list = room_list[0:num_training]
    elif testing:
        room_list = room_list[num_training+1:12]

    for room in room_list:

        # Load the scenenet room and label its objects with category ids based on the nyu mapping
        label_mapping = bproc.utility.LabelIdMapping.from_csv(bproc.utility.resolve_resource(os.path.join('id_mappings', 'nyu_idset.csv')))
        objs = bproc.loader.load_scenenet(os.path.join(args.scene_net_path, room_category, room), args.scene_texture_path, label_mapping)

        # Load all recommended cc materials, however don't load their textures yet
        cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, preload=True)

        # In some scenes floors, walls and ceilings are one object that needs to be split first
        # Collect all walls
        walls = bproc.filter.by_cp(objs, "category_id", label_mapping.id_from_label("wall"))
        # Extract floors from the objects
        new_floors = bproc.object.extract_floor(walls, new_name_for_object="floor", should_skip_if_object_is_already_there=True)
        # Set category id of all new floors
        for floor in new_floors:
            #Helper.ClearAllObjectsSceneNet(objs, cc_materials)
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


        #TODO consider increasing value

        # Color of ceiling should be emissive in different colors
        light_color = np.random.uniform(np.array([255, 241, 224, 255])/255, np.array([224, 241, 255, 255])/255)
        bproc.lighting.light_surface(ceilings, emission_strength= np.random.uniform(3,5), emission_color=light_color)

        obj_all = objs+objs_edge
        # Init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(obj_all)

        floor_materials = []
        wood = []
        for mat in cc_materials:
            name = mat.get_name()
            if 'Wood' in name or name.startswith('Bricks') or name.startswith('Carpet') or \
                    name.startswith('Concrete') or name.startswith('Paving') or \
                    name.startswith('Planks') or name.startswith('Tiles'):
                floor_materials.append(mat)
            if 'WoodFloor' in name:
                wood.append(mat)

        for obj in additional_floor:
            for i in range(len(obj.get_materials())):
                if np.random.uniform(0, 1) <= 0.6:
                    obj.set_material(i, random.choice(wood))

                elif np.random.uniform(0, 1) <= 0.8:
                    obj.set_material(i, random.choice(floor_materials))

                elif np.random.uniform(0, 1) <= 0.5:
                    # Replace the material with a random one from cc materials
                    obj.set_material(i, random.choice(cc_materials))

        # Go through all objects
        for obj in objs:

            # set material to cc_materials
            for i in range(len(obj.get_materials())):
                # In 30% of all cases
                if np.random.uniform(0, 1) <= 0.3:
                    # Replace the material with a random one from cc materials
                    obj.set_material(i, random.choice(cc_materials))



            if obj.get_name() == "carpet":
                obj.enable_rigidbody(active=False, collision_margin=0.0025)

            id = obj.get_cp("category_id")
            # find id mapping: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/resources/id_mappings/nyu_idset.csv
            ids = [2, 3, 4, 5, 6, 7, 10, 14, 15, 17, 18, 20, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40]

            id_check = id in ids

            # don't write 1. letter: gross / kleinschreibung
            if id_check:
                if not ('eco' in obj.get_name() or 'eiling' in obj.get_name() or
                        'amp' in obj.get_name() or 'ight' in obj.get_name() or 'indow' in obj.get_name()
                        or 'faucet' in obj.get_name() or 'all' in obj.get_name() or 'urtain' in obj.get_name() or
                        'alcony' in obj.get_name() or 'oor' in obj.get_name()):
                    if id == 2:
                        obj.enable_rigidbody(active=False, collision_margin=0.002)
                    else:
                        obj.enable_rigidbody(active=False, collision_margin=0.001)


            # Set all Objects in a Collection: used for render layer
            bpy.data.collections['Objects'].objects.link(bpy.data.objects[obj.get_name()])
            bpy.data.scenes['Scene'].collection.objects.unlink(bpy.data.objects[obj.get_name()])

            # delete all floor
            if obj.get_name().startswith('floor'):
                object = bpy.data.objects[obj.get_name()]
                mesh_name = object.data.name
                try:
                    bpy.data.objects.remove(object)
                    bpy.data.meshes.remove(bpy.data.meshes[mesh_name])
                    objs.remove(obj)
                except:
                    print("object or mesh has already been removed")


        # Now load all textures of the materials that were assigned to at least one object
        bproc.loader.load_ccmaterials(args.cc_material_path, fill_used_empty_materials=True)

        # set mapping of floor
        for obj in additional_floor:
            try:
                for mat_name in bpy.data.objects['Cube_Cube.001'].material_slots.keys():
                    bpy.data.materials[mat_name].node_tree.nodes["Mapping"].inputs[3].default_value = np.array([50, 50, 1])
            except:
                print("Material has no mapping")

        # Reset all category id's:
        for obj in objs:
            obj.set_cp("category_id", 0)

        # used to save images
        poses = 0
        tries = 0

        while tries < 10000 and poses < num_poses:

            # randomize material
            Helper.material_randomizer(material_color, material_pos)

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

            num = 17
            hit = np.zeros(num)
            for i in range(num-1):
                hit[i], _, _, _, _, _ = bproc.object.scene_ray_cast(location, [np.cos(i*2*np.pi/hit.shape[0]), np.sin(i*2*np.pi/hit.shape[0]), 0], hit_radius)
            hit[num-1], _, _, _, _, _ = bproc.object.scene_ray_cast(location, [0, 0, 1], 1)

            if hit.any():
                continue

            # Define a function that samples 6-DoF poses
            cube_dim = np.random.triangular(sample_cube_dims[0], sample_cube_dims[1], sample_cube_dims[2])
            loc1 = location + [cube_dim, cube_dim, 0.8]
            loc2 = location - [cube_dim, cube_dim, 0.8]

            def sample_pose(obj: bproc.types.MeshObject):
                obj.set_location(np.random.uniform(loc1, loc2))
                obj.set_rotation_euler(bproc.sampler.uniformSO3())

            # Sample the poses of all tracked objects.
            try:
                bproc.object.sample_poses(objs_edge, sample_pose_func=sample_pose)
            except:
                continue

            bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=30, check_object_interval=1)

            cam = 1
            while cam <= num_camera:

                # Sample location
                LocationCamera = bproc.sampler.shell(center=location_floor,
                                               radius_min=camera_radius[0],
                                               radius_max=camera_radius[1],
                                               elevation_min=25,
                                               elevation_max=89)
                # Compute rotation based on vector going from location towards poi
                rotation_matrix = bproc.camera.rotation_from_forward_vec(location_floor - LocationCamera, inplane_rot=np.random.uniform(-0.7854, 0.7854))

                # Add homog cam pose based on location an rotation
                cam2world_matrix = bproc.math.build_transformation_mat(LocationCamera, rotation_matrix)

                if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bvh_tree):
                    continue

                # TODO consider adding check that Camera is not outside of the wall

                bproc.camera.add_camera_pose(cam2world_matrix)
                cam += 1

            poses += 1

            #render the whole pipeline
            fx = np.random.uniform(fx_range[0], fx_range[1])
            fy = np.random.uniform(fy_range[0], fy_range[1])
            cx = np.random.uniform(cx_range[0], cx_range[1])
            cy = np.random.uniform(cy_range[0], cy_range[1])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Rendering Images at whole Resolution
            bproc.camera.set_intrinsics_from_K_matrix(K, int(image_width), int(image_height))
            bproc.renderer.set_max_amount_of_samples(samples)
            bpy.data.scenes['Scene'].cycles.use_denoising = True
            bpy.data.scenes['Scene'].view_settings.view_transform = 'Filmic'
            data = bproc.renderer.render()

            value = np.amax(np.array(data["colors"][0]), axis = -1)
            if np.mean(value) >= 80:

                # Set rendering, such that computation is faster
                bproc.renderer.set_max_amount_of_samples(1)
                bpy.data.scenes['Scene'].cycles.use_denoising = False
                bpy.data.scenes['Scene'].view_settings.view_transform = 'Standard'

                # LineArt at whole resolution
                bpy.data.scenes['Scene'].view_layers['edge'].layer_collection.children['Objects'].exclude = False
                bpy.data.scenes['Scene'].view_layers['ViewLayer'].layer_collection.children['LineArt'].exclude = False
                bpy.ops.object.lineart_bake_strokes_all()
                bpy.data.scenes['Scene'].view_layers['edge'].layer_collection.children['Objects'].exclude = True
                bpy.data.scenes['Scene'].view_layers['ViewLayer'].layer_collection.children['LineArt'].exclude = True

                # class
                for obj in objs_edge:
                    mat = bpy.data.materials[obj.get_name()]
                    # TODO
                    color = obj.get_cp("category_id") * 5 / 255.0
                    mat.grease_pencil.color = np.array([color, color, color, 1])
                bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_full['line_art_class']
                bproc.renderer.render()

                # instance
                k = 1
                for obj in objs_edge:
                    mat = bpy.data.materials[obj.get_name()]
                    color = k * 5 / 255.0
                    k += 1
                    mat.grease_pencil.color = np.array([color, color, color, 1])
                bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_full['line_art_instance']
                #bproc.renderer.render()
                seg_data_full = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])



                # Rendering Images at half Resolution
                bproc.camera.set_intrinsics_from_K_matrix(K * ratio, int(image_width * ratio), int(image_height * ratio))

                # LineArt at whole resolution
                bpy.data.scenes['Scene'].view_layers['edge'].layer_collection.children['Objects'].exclude = False
                bpy.data.scenes['Scene'].view_layers['ViewLayer'].layer_collection.children['LineArt'].exclude = False
                bpy.ops.object.lineart_bake_strokes_all()
                bpy.data.scenes['Scene'].view_layers['edge'].layer_collection.children['Objects'].exclude = True
                bpy.data.scenes['Scene'].view_layers['ViewLayer'].layer_collection.children['LineArt'].exclude = True

                # class
                for obj in objs_edge:
                    mat = bpy.data.materials[obj.get_name()]
                    color = obj.get_cp("category_id") * 5 / 255.0
                    mat.grease_pencil.color = np.array([color, color, color, 1])
                bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_half['line_art_class']
                bproc.renderer.render()

                # instance
                k = 1
                for obj in objs_edge:
                    mat = bpy.data.materials[obj.get_name()]
                    color = k * 5 / 255.0
                    k += 1
                    mat.grease_pencil.color = np.array([color, color, color, 1])
                bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = paths_half['line_art_instance']
                #bproc.renderer.render()
                seg_data_half = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])

                try:
                    if not testing_mode:
                        _ = Helper.save_data(data, seg_data_full, paths_full, names, img_idx, segmenter, color_mapping, testing_mode)
                    #img_idx = Helper.save_data(data, seg_data_half, paths_half, names, img_idx, segmenter, color_mapping)
                    img_idx = Helper.save_data(data, seg_data_half, paths_half, names, img_idx, segmenter, color_mapping, testing_mode)
                except:
                    print("Couldn't save data, Will skip it once")

            else:
                print("Images are too dark")
        if not testing_mode:
            Helper.ClearAllObjectsSceneNet(objs)

end = time.time()
with open('time.txt', 'w') as f:
    f.write('Have Computed {} images and used therefore {} '.format(poses*num_camera, end-start))


import blenderproc as bproc
import bpy
import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.colors


def get_material_color():
    """

    :return: the color parameters of the tracked materials
    """
    color = []
    pos = []

    # Material 1:
    color_array = np.ones((3, 4))
    pos_array = np.zeros((3, 1))
    for j in range(3):
        # color
        color_array[j, :] = np.array(bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].color)
        pos_array[j, :] = np.array(bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].position)
    color.append(color_array)
    pos.append(pos_array)

    # Material 2:
    color_array = np.zeros((2, 3, 4))
    pos_array = np.zeros((2, 3, 1))
    for k in range(1, 3):
        for j in range(3):
            # color
            color_array[k - 1, j, :] = \
                bpy.data.materials['2'].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].color
            pos_array[k - 1, j, :] = \
                bpy.data.materials['2'].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].position

    color.append(color_array)
    pos.append(pos_array)

    color_array = np.zeros((3, 4))
    pos_array = np.zeros((3, 1))
    for j in range(3):
        # color
        color_array[j, :] = np.array(
            bpy.data.materials['10'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].color)
        pos_array[j, :] = np.array(
            bpy.data.materials['10'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].position)
    color.append(color_array)
    pos.append(pos_array)
    return color, pos


def material_randomizer(material_color: list, material_pos: list) -> None:
    """
    Set Random Material in given range for the tracked objects

    :param material_color: Color of the Material extracted with get material
    :param material_pos: Color mixing variable extracted with get material
    """
    # Color
    color_diff_hue = 0.02
    color_diff_sat = 0.05
    color_diff_val = 0.05
    pos_diff = 0.05

    # Material Base Block
    if np.random.uniform(0, 1) <= 0.5:
        numerate = np.arange(10, 19)
    else:
        numerate = np.arange(18, 9, -1)

    color_pattern = []
    for i in numerate:
        mat_base = bpy.data.materials[str(i)].node_tree

        repeat = np.random.uniform(0, 1) <= 0.3

        # set new values
        if i == numerate[0] or not repeat:
            color1 = matplotlib.colors.rgb_to_hsv(material_color[2][0, 0:3])
            color2 = matplotlib.colors.rgb_to_hsv(material_color[2][1, 0:3])
            color3 = matplotlib.colors.rgb_to_hsv(material_color[2][2, 0:3])
            color1 = np.random.uniform(color1 - np.array([color_diff_hue, color_diff_sat, color_diff_val]),
                                       color1 + np.array([color_diff_hue, color_diff_sat, color_diff_val]))
            color2 = np.random.uniform(color2 - np.array([color_diff_hue, color_diff_sat, color_diff_val]),
                                       color2 + np.array([color_diff_hue, color_diff_sat, color_diff_val]))
            color3 = np.random.uniform(color3 - np.array([color_diff_hue, color_diff_sat, color_diff_val]),
                                       color3 + np.array([color_diff_hue, color_diff_sat, color_diff_val]))
            position1 = material_pos[2][0, :]
            position2 = material_pos[2][1, :]
            position3 = material_pos[2][2, :]
            position1 = np.random.uniform(position1 - pos_diff, position1 + pos_diff)
            position2 = np.random.uniform(position2 - pos_diff, position2 + pos_diff)
            position3 = np.random.uniform(position3 - pos_diff, position3 + pos_diff)
            musgrave_scale = np.random.uniform(2, 5)
            musgrave_lacunarity = np.random.uniform(0.5, 1.5)
            mapping_scale = np.random.triangular(0, 0.1, 1)
            noise_scale = np.random.uniform(1, 15)
            noise_roughness = np.random.uniform(0.6, 1)

        # set Color and texture
        mat_base.nodes['ColorRamp'].color_ramp.elements[0].color = np.append(matplotlib.colors.hsv_to_rgb(color1), 1)
        mat_base.nodes['ColorRamp'].color_ramp.elements[1].color = np.append(matplotlib.colors.hsv_to_rgb(color2), 1)
        mat_base.nodes['ColorRamp'].color_ramp.elements[2].color = np.append(matplotlib.colors.hsv_to_rgb(color3), 1)
        mat_base.nodes['ColorRamp'].color_ramp.elements[0].position = position1
        mat_base.nodes['ColorRamp'].color_ramp.elements[1].position = position2
        mat_base.nodes['ColorRamp'].color_ramp.elements[2].position = position3

        mat_base.nodes['Musgrave Texture'].inputs['Scale'].default_value = musgrave_scale
        mat_base.nodes['Musgrave Texture'].inputs['Lacunarity'].default_value = musgrave_lacunarity
        mat_base.nodes['Mapping'].inputs['Scale'].default_value[1] = mapping_scale
        mat_base.nodes['Noise Texture'].inputs['Scale'].default_value = noise_scale
        mat_base.nodes['Noise Texture'].inputs['Roughness'].default_value = noise_roughness

    # Material Bars
    for i in range(2, 10):
        # Color
        for j in range(3):
            for k in range(1, 3):
                # color
                color = matplotlib.colors.rgb_to_hsv(material_color[1][k - 1, j, 0:3])
                color = np.random.uniform(color - np.array([color_diff_hue, color_diff_sat, color_diff_val]),
                                          color + np.array([color_diff_hue, color_diff_sat, color_diff_val]))
                bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].color = \
                    np.append(matplotlib.colors.hsv_to_rgb(color), 1)

                # color pos:
                position = material_pos[1][k - 1, j, :]
                bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].position = \
                    np.random.uniform(position - pos_diff, position + pos_diff)

        # texture
        mat_bar = bpy.data.materials[str(i)].node_tree
        mat_bar.nodes['Wave Texture'].inputs['Scale'].default_value = np.random.uniform(2, 8)
        mat_bar.nodes['Wave Texture'].inputs['Distortion'].default_value = np.random.uniform(3, 8)
        mat_bar.nodes['Mix'].inputs['Fac'].default_value = np.random.uniform(0.2, 0.4)
        mat_bar.nodes['Mapping.001'].inputs['Rotation'].default_value = np.random.uniform([0, 0, 0],
                                                                                          np.pi / 2 * np.array(
                                                                                              [1, 1, 1]))
        mat_bar.nodes['Mapping.001'].inputs['Scale'].default_value = np.random.uniform([8, 0, 8], [10, 0.2, 10])

    # Material plates
    bpy.data.materials['1'].node_tree.nodes['Noise Texture'].inputs['Roughness'].default_value = np.random.uniform(0.92,
                                                                                                                   1)
    for j in range(3):
        # color
        color = matplotlib.colors.rgb_to_hsv(material_color[0][j, 0:3])
        color = np.random.uniform(color - np.array([color_diff_hue, color_diff_sat, color_diff_val]),
                                  color + np.array([color_diff_hue, color_diff_sat, color_diff_val]))
        bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].color = \
            np.append(matplotlib.colors.hsv_to_rgb(color), 1)

        # color pos:
        position = material_pos[0][j, :]
        bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].position = \
            np.random.uniform(position - pos_diff, position + pos_diff)


def save_data(data: dict, seg_data: dict, paths: dict, names: list, img_idx: int, segmenter: bool, color_mapping: dict,
              testing_mode: bool) -> int:
    """
    Function to save all rendered images to the right structure and remap color values used

    :param data: dictionary containing the color images, output of bproc.render
    :param seg_data: dictionary containing the segmentation map, output of bproch.render
    :param paths: dictionary containing all relevant paths to store everything
    :param names: list of all tracked object names
    :param img_idx: current img_idx used to store next image, hdf5 container, coco
    :param segmenter: boolean to decide if segmentation map or edge segmentation is stored
    :param color_mapping: dictionary to define colormapping, as color values defined in blender don't correspond to the ones stored in the png images
    :param testing_mode: boolean to define if we run a test mode: only one rendering
    :return: new img_idx
    """
    # Use Freestyle Image information together with segmentation to get semantic edges ground truth
    class_segmaps = seg_data["class_segmaps"]
    instance_segmaps = seg_data["instance_segmaps"]
    colors = data["colors"]
    attribute_maps = seg_data["instance_attribute_maps"]
    num_frames = len(colors)

    new_attribute_maps = []
    mapping = []

    # TODO: Only run segmentation if required:  adaptive number of inputs into function,
    #                                           check which elements are in image,
    #                                           simplify new attribute maps

    if len(colors) == len(class_segmaps):

        for i in range(num_frames):
            new_attribute_maps.append([])
            mapping.append([])
            for dict in attribute_maps[i]:
                for idx in range(len(names)):
                    if dict["name"] == names[idx]:
                        new_attribute_maps[i].append(
                            {"idx": idx, "category_id": dict["category_id"], "name": dict["name"]})
                        mapping[i].append({"old_idx": dict["idx"], "new_idx": idx})

        # Save in CaseNet format
        for i in range(num_frames):

            if not segmenter:
                class_image = Image.open(paths['line_art_class'] + '/{:04d}.png'.format(i))
                class_image = np.array(class_image, dtype=np.uint8)
                class_image = np.where(class_image <= 10, 0, class_image)
                num_classes = 3
                for cat in range(1, num_classes + 1):
                    class_image = np.where((class_image <= color_mapping[str(cat)] + 1) &
                                           (class_image >= color_mapping[str(cat)] - 1), cat, class_image)

                class_image = np.where(class_image >= num_classes + 1, 0, class_image)

                # print(class_image)
                class_segmaps[i] = class_image
            class_segmaps[i] = class_segmaps[i].astype(np.uint8)

            if testing_mode:
                class_segmaps_image = Image.fromarray(class_segmaps[i] * 80)
            else:
                class_segmaps_image = Image.fromarray(class_segmaps[i] * 80)
            class_segmaps_image.save(paths["class_annotation"] + '/{:04d}.png'.format(img_idx))

            if not segmenter:
                instance_image = Image.open(paths['line_art_instance'] + '/{:04d}.png'.format(i))
                instance_image = np.array(instance_image, dtype=np.uint8)
                instance_image = np.where(instance_image <= 10, 0, instance_image)
                for inst in range(1, len(names) + 1):
                    instance_image = np.where((instance_image <= color_mapping[str(inst)] + 1) &
                                              (instance_image >= color_mapping[str(inst)] - 1), inst, instance_image)

                instance_image = np.where(instance_image >= len(names) + 1, 0, instance_image)
                instance_segmaps[i] = instance_image
                # print(instance_image)
            else:
                instance_segmaps[i] = instance_segmaps[i].astype(np.uint8)
                # perform mapping
                for maps in mapping[i]:
                    instance_segmaps[i] = np.where(
                        (instance_segmaps[i] <= maps["old_idx"] + 0.5) & (instance_segmaps[i] >= maps["old_idx"] - 0.5),
                        maps["new_idx"], instance_segmaps[i])
            instance_segmaps[i] = instance_segmaps[i].astype(np.uint8)
            if testing_mode:
                instance_segmaps_image = Image.fromarray(instance_segmaps[i] * 15)
            else:
                instance_segmaps_image = Image.fromarray(instance_segmaps[i] * 15)
            instance_segmaps_image.save(paths["instance_annotation"] + '/{:04d}.png'.format(img_idx))

            color_image = Image.fromarray(np.array(colors[i], dtype=np.uint8))
            color_image.save(paths["images"] + '/{:04d}.png'.format(img_idx))

            print("\n \n \n Save Image {} \n \n \n ".format(img_idx))

            img_idx += 1

        data["class_segmaps"] = class_segmaps
        data["instance_segmaps"] = instance_segmaps
        data["instance_attribute_maps"] = new_attribute_maps

        bproc.writer.write_hdf5(paths["hdf5"], data, append_to_existing_output=True)

        bproc.writer.write_coco_annotations(
            paths["coco"],
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            append_to_existing_output=True,
            mask_encoding_format="rle",
            color_file_format="PNG")

    else:
        print("error occured during image segmentation in BlenderProc")

    return img_idx


def clean_output(paths: dict) -> None:
    """
    Delete the output of an old data rendering run. Only used if append_to_existing = False

    :param paths: dictionary containing all relevant paths in which data is stored
    """
    for path in paths.values():
        shutil.rmtree(path)

    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)


def get_idx(paths: dict) -> int:
    """

    :param paths: dictionary containing all relevant paths in which data is stored
    :return: current largest index of the stored images
    """
    img_idx = 0
    # Look for image with highest index
    for path in os.listdir(paths["images"]):
        if path.endswith(".png"):
            index = path[:-len(".png")]
            if index.isdigit():
                img_idx = max(img_idx, int(index) + 1)
    return img_idx


def ClearAllObjectsSceneNet(objs: list) -> None:
    """
    Delete all materials and objects not used by the tracked objects or the additonal floor

    :param objs: list of all objects loaded from the SceneNet
    """
    for o in objs:
        object = bpy.data.objects[o.get_name()]
        mesh_name = object.data.name
        try:
            bpy.data.objects.remove(object)
            bpy.data.meshes.remove(bpy.data.meshes[mesh_name])
        except:
            print("object or mesh has already been removed")

    for m in bpy.data.materials:
        if m.users == 0:
            bpy.data.materials.remove(m)

    for img in bpy.data.images:
        bpy.data.images.remove(img)
    try:
        for mat_name in bpy.data.objects['Cube_Cube.001'].material_slots.keys():
            bpy.data.materials.remove(bpy.data.materials[mat_name])
    except:
        print("\n \n \n \n No material was assigned to additonal floor")

import blenderproc as bproc
import bpy
import numpy as np
from PIL import Image
import os
import shutil


def material_randomizer():

    # Color
    color_diff = 0.03

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

            color1 = mat_base.nodes['ColorRamp'].color_ramp.elements[0].color
            color2 = mat_base.nodes['ColorRamp'].color_ramp.elements[1].color
            color3 = mat_base.nodes['ColorRamp'].color_ramp.elements[2].color
            color1 = np.random.uniform(color1 - color_diff * np.array([1, 1, 1, 1]),
                                       color1 + color_diff * np.array([1, 1, 1, 1]))
            color2 = np.random.uniform(color2 - color_diff * np.array([1, 1, 1, 1]),
                                       color2 + color_diff * np.array([1, 1, 1, 1]))
            color3 = np.random.uniform(color3 - color_diff * np.array([1, 1, 1, 1]),
                                       color3 + color_diff * np.array([1, 1, 1, 1]))
            position1 = mat_base.nodes['ColorRamp'].color_ramp.elements[0].position
            position2 = mat_base.nodes['ColorRamp'].color_ramp.elements[1].position
            position3 = mat_base.nodes['ColorRamp'].color_ramp.elements[2].position
            position1 = np.random.uniform(position1-0.1,position1+0.1)
            position2 = np.random.uniform(position2 - 0.1, position2 + 0.1)
            position3 = np.random.uniform(position3 - 0.1, position3 + 0.1)
            musgrave_scale = np.random.uniform(2, 5)
            musgrave_lacunarity = np.random.uniform(0.5, 1.5)
            mapping_scale = np.random.uniform(0, 2)
            noise_scale = np.random.uniform(1, 15)
            noise_roughness = np.random.uniform(0.6, 1)

        # set Color and texture
        mat_base.nodes['ColorRamp'].color_ramp.elements[0].color = color1
        mat_base.nodes['ColorRamp'].color_ramp.elements[1].color = color2
        mat_base.nodes['ColorRamp'].color_ramp.elements[2].color = color3
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
                color = bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].color
                bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].color =\
                    np.random.uniform(color - color_diff * np.array([1, 1, 1, 1]), color + color_diff * np.array([1, 1, 1, 1]))
                # color pos:
                position = bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].position
                bpy.data.materials[str(i)].node_tree.nodes['ColorRamp{}'.format(k)].color_ramp.elements[j].position = \
                    np.random.uniform(position-0.1,position+0.1)

        # texture
        mat_bar = bpy.data.materials[str(i)].node_tree
        mat_bar.nodes['Wave Texture'].inputs['Scale'].default_value = np.random.uniform(2, 8)
        mat_bar.nodes['Wave Texture'].inputs['Distortion'].default_value = np.random.uniform(3, 8)
        mat_bar.nodes['Mix'].inputs['Fac'].default_value = np.random.uniform(0.2, 0.4)
        mat_bar.nodes['Mapping.001'].inputs['Rotation'].default_value = np.random.uniform([0, 0, 0], np.pi/2*np.array([1, 1, 1]))
        mat_bar.nodes['Mapping.001'].inputs['Scale'].default_value = np.random.uniform([8, 0, 8], [10, 0.2, 10])

    # Material1
    bpy.data.materials['1'].node_tree.nodes['Noise Texture'].inputs['Roughness'].default_value = np.random.uniform(0.92, 1)
    for j in range(3):
        # color
        color = bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].color
        bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].color = \
            np.random.uniform(color - color_diff * np.array([1, 1, 1, 1]), color + color_diff * np.array([1, 1, 1, 1]))
        # color pos:
        position = bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].position
        bpy.data.materials['1'].node_tree.nodes['ColorRamp'].color_ramp.elements[j].position = \
            np.random.uniform(position - 0.1, position + 0.1)


def save_data(data, seg_data, paths, names, img_idx, segmenter):

    # Use Freestyle Image information together with segmentation to get semantic edges ground truth
    class_segmaps = seg_data["class_segmaps"]
    instance_segmaps = seg_data["instance_segmaps"]
    colors = data["colors"]
    attribute_maps = seg_data["instance_attribute_maps"]
    num_frames = len(colors)

    new_attribute_maps = []
    mapping = []

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

            freestyle_image = Image.open(paths['freestyle'] + '/Freestyle{:04d}.png'.format(i))

            freestyle_image = np.array(freestyle_image, dtype=np.uint8)
            freestyle_image = np.where(freestyle_image <= 2, 0, 1)

            if not segmenter:
                class_segmaps[i] = class_segmaps[i] * freestyle_image
            class_segmaps[i] = class_segmaps[i].astype(np.uint8)
            # TODO
            class_segmaps_image = Image.fromarray(class_segmaps[i]*80)
            class_segmaps_image.save(paths["class_annotation"] + '/{:04d}.png'.format(img_idx))

            if not segmenter:
                instance_segmaps[i] = np.array(instance_segmaps[i], dtype=np.uint8) * freestyle_image
            else:
                instance_segmaps[i] = instance_segmaps[i].astype(np.uint8)
            for maps in mapping[i]:
                instance_segmaps[i] = np.where(
                    (instance_segmaps[i] <= maps["old_idx"] + 0.5) & (instance_segmaps[i] >= maps["old_idx"] - 0.5),
                    maps["new_idx"], instance_segmaps[i])
            instance_segmaps[i] = instance_segmaps[i].astype(np.uint8)
            instance_segmaps_image = Image.fromarray(instance_segmaps[i])
            instance_segmaps_image.save(paths["instance_annotation"] + '/{:04d}.png'.format(img_idx))

            color_image = Image.fromarray(np.array(colors[i], dtype=np.uint8))
            color_image.save(paths["images"] + '/{:04d}.png'.format(img_idx))

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


def clean_output(paths):
    for path in paths.values():
        shutil.rmtree(path)

    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)


def get_idx(paths):
    img_idx = 0
    # Look for image with highest index
    for path in os.listdir(paths["images"]):
        if path.endswith(".png"):
            index = path[:-len(".png")]
            if index.isdigit():
                img_idx = max(img_idx, int(index) + 1)
    return img_idx

import shutil
from typing import Dict
import numpy as np
import cv2
import os
from tqdm import tqdm
import json


SUPPORT_RAW_EXT = (".jpg", ".png", ".bmp")  # 数据集图片原始格式, jpg、png、bmp
TARGET_EXT = ".png"  # 数据集图片需要转为png


#
def convert_2_target_ext(path: str, target_ext: str = TARGET_EXT):

    # path= 'E:/TrainingFramework/DataSets/Dataset10086_boge_void_2nn/train/'

    # raw_ext = '.jpg'
    # target_ext = '.png'

    files = os.listdir(path)
    pbar = tqdm(files, total=len(files), desc="convert images to .png")

    for f in pbar:
        old_path = os.path.join(path, f)
        image = cv2.imread(old_path, cv2.IMREAD_COLOR_BGR)

        name = os.path.splitext(f)[0]
        new_path = os.path.join(path, name + target_ext)
        cv2.imwrite(new_path, image)

        os.remove(old_path)
        ...


def color2gray(path: str):

    path = "E:/TrainingFramework/nnUNet-master/DATASET/nnUNet_raw/Dataset10086_boge_void_2nn/labelsTr/"
    ext = ".png"

    files = os.listdir(path)
    pbar = tqdm.tqdm(files, total=len(files))

    for f in pbar:
        if f.endswith(ext):
            img_path = os.path.join(path, f)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(img_path, image)
            ...


def json_to_labelimage(label_path: str, class_map: Dict):

    # label_path = 'E:/TrainingFramework/nnUNet-master/DATASET/rrrr/Dataset003_Position/image/'
    label_extension = ".json"

    # 0 : bg
    """
    class_map = {
        'a':1,
        'b':2,
        'c':3,
    }
    """

    # json -> mask image
    json_files = [f for f in os.listdir(label_path) if f.endswith(label_extension)]
    pbar = tqdm(json_files, total=len(json_files), desc="convert json to png")

    label_names = []

    for i, filename in enumerate(pbar):
        cur_json_file = os.path.join(label_path, filename)
        # if file.endswith(label_extension):
        with open(cur_json_file, "r") as f:
            data = json.load(f)
            # h,w
            h = data["imageHeight"]
            w = data["imageWidth"]

            # mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # record
            areas = []
            points = []
            labels = []

            for shape in data["shapes"]:
                label = shape["label"]
                pts = shape["points"]
                area = cv2.contourArea(np.array(pts, dtype=np.float32))

                areas.append(area)
                points.append(pts)
                labels.append(label)

            areas = np.array(areas)
            index = np.argsort(areas)  # 从小到大
            index = index[::-1]  # 从大到小
            for idx in index:
                pts = np.array(points[idx], dtype=np.int32)
                mask = cv2.fillPoly(mask, [pts], class_map[labels[idx]])

            # cv2.imshow('img', mask)
            # cv2.waitKey(0)
            ...
            crop = len(label_extension)
            save_path = os.path.join(label_path, f"{filename[:-crop]}{TARGET_EXT}")
            label_names.append(save_path)
            cv2.imwrite(save_path, mask)

        # 删除json
        os.remove(cur_json_file)

    return label_names


def letter_box(image: np.ndarray, fill_value=114, pre_channel=True):
    """
    pre_channel: if True, image is (c,h,w), else (h,w,c)
    """

    assert image.ndim == 3 or image.ndim == 2  # (c,h,w) or (h,w)

    ndims = image.ndim

    # s = max(h,w)
    s = max(image.shape)  # assume h,w > c

    if ndims == 2:
        h, w = image.shape
    else:
        h, w = image.shape[1:] if pre_channel else image.shape[:-1]

    h_diff = s - h
    w_diff = s - w

    h_start = h_diff // 2
    h_stop = h_start + h

    w_start = w_diff // 2
    w_stop = w_start + w

    if pre_channel:  # (c,h,w)
        if ndims == 2:
            image = image[None, :, :]

        c, h, w = image.shape
        mask = np.ones((c, s, s), image.dtype) * fill_value
        mask[:, h_start:h_stop, w_start:w_stop] = image

    else:  # (h,w,c)
        if ndims == 2:
            image = image[..., None]

        h, w, c = image.shape
        mask = np.ones((s, s, c), image.dtype) * fill_value
        mask[h_start:h_stop, w_start:w_stop, :] = image

    return (mask[0] if pre_channel else mask[..., 0]) if ndims == 2 else mask


def letter_box_label_image(label_image: np.ndarray, pre_channel=True):
    assert label_image.ndim == 3 or label_image.ndim == 2  # (c,h,w) or (h,w)
    # 多通道时,通道0为背景; 单通道时, 像素值0为背景
    ndims = label_image.ndim

    # s = max(h,w)
    s = max(label_image.shape)  # assume h,w > c

    if ndims == 2:
        h, w = label_image.shape
    else:
        h, w = label_image.shape[1:] if pre_channel else label_image.shape[:-1]

    h_diff = s - h
    w_diff = s - w

    h_start = h_diff // 2
    h_stop = h_start + h

    w_start = w_diff // 2
    w_stop = w_start + w

    if pre_channel:
        if ndims == 2:
            label_image = label_image[None, :, :]

        c, h, w = label_image.shape

        mask = np.zeros((c, s, s), label_image.dtype)

        mask[:, h_start:h_stop, w_start:w_stop] = label_image

    else:
        if ndims == 2:
            label_image = label_image[..., None]

        h, w, c = label_image.shape

        mask = np.zeros((s, s, c), label_image.dtype)

        mask[h_start:h_stop, w_start:w_stop, :] = label_image

    if ndims != 2:
        fg_mask = mask[1:, ...].sum(0) != 0 if pre_channel else mask[..., 1:].sum(-1) != 0
        bg_mask = ~fg_mask

        if pre_channel:
            mask[0][bg_mask] = 1
        else:
            mask[..., 0][bg_mask] = 1  # instead of mask[bg_mask][0] = 1

    return (mask[0] if pre_channel else mask[..., 0]) if ndims == 2 else mask


def letterbox_and_resize_image(path: str, new_size: int = 1024, padding_value: int = 128, ext=SUPPORT_RAW_EXT):

    files = [f for f in os.listdir(path) if f.endswith(ext)]

    pbar = tqdm(files, total=len(files), desc=f"letterbox images and resize to {new_size}")

    for f in pbar:
        image_path = os.path.join(path, f)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
        square_image = letter_box(image, padding_value, pre_channel=False)  # 128
        #
        image_hwc = cv2.resize(src=square_image, dsize=(new_size, new_size), interpolation=cv2.INTER_AREA)  # AREA
        # name + target_ext
        new_path = os.path.splitext(image_path)[0] + TARGET_EXT
        cv2.imwrite(new_path, image_hwc)

        if image_path != new_path:
            os.remove(image_path)


def letterbox_and_resize_label(path: str, new_size: int = 1024, ext=".png"):

    files = [f for f in os.listdir(path) if f.endswith(ext)]

    pbar = tqdm(files, total=len(files), desc=f"letterbox label images and resize to {new_size}")

    for f in pbar:
        image_path = os.path.join(path, f)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        square_image = letter_box_label_image(image, pre_channel=False)  # 0
        #
        resized_image = cv2.resize(
            src=square_image, dsize=(new_size, new_size), interpolation=cv2.INTER_NEAREST_EXACT
        )  # NEAREST
        # name = os.path.splitext(f)[0]
        cv2.imwrite(image_path, resized_image)


# 重命名label rename
def rename_labelimage(root: str, dataset_id_name: str):
    # root = 'E:/TrainingFramework/DataSets/Dataset10086_boge_void_2nn'
    # label_folder = 'labelsTr'
    # id_name = 'boge_void_2nn'

    path2 = os.path.join(root)

    label_images = [f for f in os.listdir(path2) if f.endswith(f"{TARGET_EXT}")]

    pbar = tqdm(label_images, total=len(label_images), desc="rename label images")
    for i, filename in enumerate(pbar):
        old_name = os.path.join(path2, filename)
        new_name = os.path.join(path2, dataset_id_name + "_" + str(i) + f"{TARGET_EXT}")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)


# 重命名image rename
def rename_image(root: str, dataset_id_name: str):
    # root = 'E:/TrainingFramework/DataSets/Dataset10086_boge_void_2nn'
    # image_folder = 'imagesTr'
    # id_name = 'boge_void_2nn'

    path1 = os.path.join(root)

    images = [f for f in os.listdir(path1) if f.endswith(f"{TARGET_EXT}")]
    pbar = tqdm(images, total=len(images), desc="rename images")
    for i, filename in enumerate(pbar):
        old_name = os.path.join(path1, filename)
        new_name = os.path.join(path1, dataset_id_name + "_" + str(i) + f"_0000{TARGET_EXT}")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)


# 可视化标签图
def visualize(path: str, save_path: str):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # path = 'E:/TrainingFramework/nnUNet-master/DATASET/nnUNet_raw/Dataset10086_boge_void_2nn/labelsTr/'

    files = [f for f in os.listdir(path) if f.endswith(TARGET_EXT)]

    pbar = tqdm(files, total=len(files), desc="create visualize images")

    for filename in pbar:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        labels = np.unique(img)
        max_ = img.max()
        img = np.astype(img, np.float64)
        img = img * 255 / max_
        img = np.astype(img, np.uint8)

        cv2.imwrite(os.path.join(save_path, filename.replace(TARGET_EXT, ".jpg")), img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)


# 清空上一次转换结果
def clear(
    json_path: str,
    image_path: str,
):

    converted_label_images = [f for f in os.listdir(json_path) if f.endswith(TARGET_EXT)]
    for f in converted_label_images:
        os.remove(os.path.join(json_path, f))

    converted_images = [f for f in os.listdir(image_path) if f.endswith(TARGET_EXT)]
    for f in converted_images:
        os.remove(os.path.join(image_path, f))


# 检查图片目录格式冲突
def check_ext(image_path: str):

    images = [f for f in os.listdir(image_path)]
    pbar = tqdm(images, total=len(images), desc="检查图片格式冲突")

    cnt = 0
    for f in pbar:
        if f.endswith(TARGET_EXT):
            cnt += 1
            print(f)

    assert cnt == 0, "图片格式冲突, 先转成其他格式再继续"


def backup_(
    image_path: str,
    label_path: str,
):

    # backup image
    image_norm_path = os.path.normpath(image_path)  # dirname(path)# basename(path)
    image_result_path = image_norm_path + "_result"

    if os.path.exists(image_result_path):
        # os.chmod(image_result_path, 0o777)  # 赋予读写权限
        # os.remove(backup_path)
        shutil.rmtree(image_result_path)
    shutil.copytree(image_norm_path, image_result_path)

    # backup label
    label_norm_path = os.path.normpath(label_path)  # dirname(path)# basename(path)
    label_result_path = label_norm_path + "_result"

    if os.path.exists(label_result_path):
        # os.chmod(label_result_path, 0o777)  # 赋予读写权限
        # os.remove(backup_path)
        shutil.rmtree(label_result_path)
    shutil.copytree(label_norm_path, label_result_path)

    return image_result_path, label_result_path


def main_process(
    json_path: str,
    image_path: str,
    class_map: Dict,
    new_size: int = 1024,
    label_padding_value: int = 0,
    image_padding_value: int = 128,
    dataset_id_name: str = "",
):
    # data_path : 初始目录, 包含labelme标注文件(json和.jpg文件)

    assert json_path != image_path, " 把json文件和图片文件放到不同的目录中! "

    # 0. clear
    # clear(json_path=json_path, image_path=image_path)

    # 1. json2labelimage, '.json' -> '.png'
    label_names = json_to_labelimage(json_path, class_map)

    # 2. label image, letterbox and resize to 1024
    letterbox_and_resize_label(json_path, new_size, TARGET_EXT)

    # 3. image, letterbox and resize
    # convert_2_target_ext(image_path) # jpg -> png if any.
    letterbox_and_resize_image(
        image_path,
        new_size,
        image_padding_value,
    )

    # 4. rename image
    rename_image(image_path, dataset_id_name)

    # 5. rename label image
    rename_labelimage(json_path, dataset_id_name)


if __name__ == "__main__":
    ###################################################################
    #
    #
    #
    ###################################################################
    # 原始目录
    json_path = "E:/TrainingFramework/nnUNet-master/DATASET/rrrr/Dataset515_location/labels/"
    image_path = "E:/TrainingFramework/nnUNet-master/DATASET/rrrr/Dataset515_location/images/"
    visualize_path = "E:/TrainingFramework/nnUNet-master/DATASET/rrrr/Dataset515_location/viz"
    # 数据集名称
    dataset_id_name = "Dataset515_bogepos"
    # 训练图片尺寸
    new_size = 1024

    # 如果不需要某个标签, 请注释掉该标签
    class_map = {
        "bg": 0,
        "a": 1,
        # 'b' : 2,
        # 'c' : 3,
    }

    ###################################################################
    #
    #
    #
    ###################################################################
    # unchanged
    label_padding_value = 0
    image_padding_value = 128

    # 先检查是否有图片格式冲突, 原始图片格式不能与TARGET_EXT相同
    # check_ext(image_path)

    image_result_path, label_result_path = backup_(image_path, json_path)

    # 开始处理
    main_process(
        json_path=label_result_path,
        image_path=image_result_path,
        class_map=class_map,
        new_size=new_size,
        label_padding_value=label_padding_value,
        image_padding_value=image_padding_value,
        dataset_id_name=dataset_id_name,
    )

    # 可视化标签
    visualize(label_result_path, visualize_path)

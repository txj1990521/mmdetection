import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split
from labelme import utils

np.random.seed(41)
# 0为背景
classname_to_id = {
    '出血': 1,
}


# 注意这里：yxf
# 需要从1开始把对应的Label名字写入：这里根据自己的Lable名字修改

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}

        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace("json", obj['imagePath'].split('.')[-1])
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    # 这里注意：yxf
    # 需要把labelme_path修改为自己放images和json文件的路径
    labelme_path = "Z:/病灶识别第一批（已审核）/出血01-13/胶囊/经典"
    saved_coco_path = "D:/tmpdata"
    # 要把saved_coco_path修改为自己放生成COCO的路径，这里会在我当前COCO的文件夹下建立生成coco文件夹。
    print('reading...')
    # 创建文件
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/images/train2017/" % saved_coco_path):
        os.makedirs("%scoco/images/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/images/val2017/" % saved_coco_path):
        os.makedirs("%scoco/images/val2017" % saved_coco_path)
    # 获取images目录下所有的json文件列表
    print(labelme_path + "/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    image_list_path = []
    imagetype = ['.jpg', 'png', 'jpeg']
    for x in imagetype:
        image_list_tmp = glob.glob(labelme_path + "/*"+x)
        image_list_path.extend(image_list_tmp)
    print('json_list_path: ', len(json_list_path))
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.1, train_size=0.9)

    # 这里yxf：将训练集和验证集的比例是9：1，可以根据自己想要的比例修改。
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式

    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    # for file in train_path:
    #     img_name = file.replace('json', 'png')
    #     temp_img = cv2.imdecode(np.fromfile(img_name.replace('\\', '/'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    #     # temp_img = cv2.imread(img_name)
    #     # print(temp_img) 测试图像读取是否正确
    #     try:
    #         img_name_jpg = img_name.replace('png', 'jpg')
    #         print("jpg测试:" + img_name_jpg)
    #         filenames = img_name_jpg.split("\\")[-1]
    #         print(filenames)  # 这里是将一个路径中的文件名字提取出来
    #         cv2.imwrite("./COCO/coco/images/train2017/{}".format(filenames), temp_img)
    #         # 这句写入语句，是将 X.jpg 写入到指定路径./COCO/coco/images/train2017/X.jpg
    #     except Exception as e:
    #         print(e)
    #         print('Wrong Image:', img_name)
    #         continue
    #
    #     print(img_name + '-->', img_name.replace('png', 'jpg'))
    #     # print("yxf"+img_name)
    #
    # for file in val_path:
    #     # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)
    #
    #     img_name = file.replace('json', 'png')
    #     temp_img = cv2.imread(img_name)
    #     try:
    #
    #         # cv2.imwrite("{}coco/images/val2017/{}".format(saved_coco_path, img_name.replace('png', 'jpg')), temp_img)
    #         img_name_jpg = img_name.replace('png', 'jpg')  # 将png文件替换成jpg文件。
    #         print("jpg测试:" + img_name_jpg)
    #         filenames = img_name_jpg.split("\\")[-1]
    #         print(filenames)
    #         cv2.imwrite("./COCO/coco/images/val2017/{}".format(filenames), temp_img)
    #     except Exception as e:
    #         print(e)
    #         print('Wrong Image:', img_name)
    #         continue
    #     print(img_name + '-->', img_name.replace('png', 'jpg'))

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

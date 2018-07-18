import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from sklearn import model_selection
from sklearn.cross_validation import StratifiedShuffleSplit
from torch.utils import data

from transform import HorizontalFlip, VerticalFlip


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def default_loader(path):
    return Image.open(path)


class CityDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True,
                 label_type=None, input_ch=3):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:


        # import glob
        # input_filenames = glob.glob(os.path.join(root + 'images/', '*.png'))
        #
        # file_write = open(root + 'train.txt','w')
        # for filename in input_filenames:
        #     file_write.write('%s\n' % filename)
        #
        # file_write.close()
        #
        # file_val = open(root + 'val.txt','w')
        # for filename in input_filenames:
        #     file_val.write('%s\n' % filename)
        #
        # file_val.close()

        # input_filenames = glob.iglob(root + 'leftImg8bit/train/' + '/**/*.c', recursive=True)

        input_filenames = []
        for imdir, dirnames, filenames in os.walk(os.path.join(root, 'leftImg8bit/train/')):
            input_filenames.extend(glob.glob(imdir + "/*.png"))

        file_write = open(root + 'leftImg8bit/' + 'train.txt','w')
        label_file_write = open(root + 'leftImg8bit/' + 'train_label.txt', 'w')
        for filename in input_filenames:
            file_write.write('%s\n' % filename)

            filename = filename.replace('leftImg8bit.png', 'gtFine_labelIds.png')
            filename = filename.replace('leftImg8bit/', 'gtFine/')
            label_file_write.write('%s\n' % filename)
        file_write.close()
        label_file_write.close()

        val_filenames = []
        for imdir, dirnames, filenames in os.walk(os.path.join(root, 'leftImg8bit/val/')):
            val_filenames.extend(glob.glob(imdir + "/*.png"))


        file_val = open(root + 'leftImg8bit/' + 'val.txt','w')
        label_file_val = open(root + 'leftImg8bit/' + 'val_label.txt', 'w')
        for filename in val_filenames:
            file_val.write('%s\n' % filename)

            filename = filename.replace('leftImg8bit.png', 'gtFine_labelIds.png')
            filename = filename.replace('leftImg8bit/', 'gtFine/')
            label_file_val.write('%s\n' % filename)

        file_val.close()
        label_file_val.close()

        # import ipdb; ipdb.set_trace()

        for split in ['train','val']:
            label_filenames = []
            imgsets_dir = osp.join(data_dir, "leftImg8bit/%s.txt" % split)
            with open(imgsets_dir) as imgset_file:
                for name in imgset_file:
                    name = name.strip()
                    # img_file = osp.join(data_dir, "leftImg8bit/%s" % name)
                    img_file = name
                    if label_type == "label16":
                        name = name.replace('leftImg8bit.png', 'gtFine_label16IDs.png')
                        name = name.replace('leftImg8bit/', 'gtFine/')
                    else:
                        name = name.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                        name = name.replace('leftImg8bit/', 'gtFine/')

                    # label_file = osp.join(data_dir, "gtFine/%s" % name)
                    label_file = name
                    label_filenames.extend(name)
                    self.files[split].append({
                        "img": img_file,
                        "label": label_file
                    })


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class GTADataSet(data.Dataset):
    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # Note; split "train" and "images" are SAME!!!

        assert split in ["images", "test", "train"]

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root

        import glob
        input_filenames = glob.glob(os.path.join(root + 'images/', '*.png'))

        # TODO(ajaytanwani): add only 70 percent of images
        file_write = open(root + 'train.txt','w')
        for filename in input_filenames:
            file_write.write('%s\n' % filename)

        file_write.close()

        # TODO(ajaytanwani): add remaining 30 percent of images
        file_val = open(root + 'val.txt','w')
        for filename in input_filenames:
            file_val.write('%s\n' % filename)

        file_val.close()

        imgsets_dir = osp.join(data_dir, "%s.txt" % split)

        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "%s" % name)
                # name = name.replace('leftImg8bit','gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, split="all", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # TODO this does not support "split" parameter

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.test = test

        rgb_dir = osp.join(root, "RGB")
        gt_dir = osp.join(root, "GT", "LABELS16")

        rgb_fn_list = glob.glob(osp.join(rgb_dir, "*.png"))
        gt_fn_list = glob.glob(osp.join(gt_dir, "*.png"))

        for rgb_fn, gt_fn in zip(rgb_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "label": gt_fn
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        img_file = datafiles["rgb"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class TestDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3):
        assert input_ch == 3
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.listdir(data_dir)
        for name in imgsets_dir:
            img_file = osp.join(data_dir, "%s" % name)
            self.files[split].append({
                "img": img_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return img, 'hoge', img_file
        else:
            return img, img


class RealRobotDataSet():
    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # Note; split "train" and "images" are SAME!!!

        # assert split in ["image_rgb", "image_depth"]

        # assert input_ch in [1, 2, 3]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        test_split_ratio = 0.2

        import glob
        image_filenames = glob.glob(os.path.join(root + 'image_rgb/', '*.png'))
        image_filenames.sort()

        label_filenames = glob.glob(os.path.join(root + 'image_labels/', '*.png'))
        label_filenames.sort()

        train_images_ids, test_images_ids = model_selection.train_test_split(np.arange(len(image_filenames)), test_size=test_split_ratio, random_state=0)

        train_images_filenames = [image_filenames[j] for j in train_images_ids]
        test_images_filenames = [image_filenames[j] for j in test_images_ids]

        train_labels_filenames = [label_filenames[j] for j in train_images_ids]
        test_labels_filenames = [label_filenames[j] for j in test_images_ids]

        file_write = open(root + 'train.txt','w')
        file_label_write = open(root + 'train_label.txt','w')
        for train_filename, label_filename in zip(train_images_filenames, train_labels_filenames):
            file_write.write('%s\n' % train_filename)
            file_label_write.write('%s\n' % label_filename)

            self.files[split].append({
                "img": train_filename,
                "label": label_filename,
            })

        file_write.close()
        file_label_write.close()

        file_write = open(root + 'val.txt','w')
        file_label_write = open(root + 'val_label.txt','w')
        for test_filename, test_label_filename in zip(test_images_filenames, test_labels_filenames):
            file_write.write('%s\n' % test_filename)
            file_label_write.write('%s\n' % test_label_filename)
        file_write.close()
        file_label_write.close()

    # def __init__(self, root, split="images", img_transform=None, label_transform=None,
    #              test=False, input_ch=3):
    #     # Note; split "train" and "images" are SAME!!!
    #
    #     assert split in ["images", "test", "train"]
    #
    #     assert input_ch in [1, 3, 4]
    #     self.input_ch = input_ch
    #     self.root = root
    #     self.split = split
    #     # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    #     self.files = collections.defaultdict(list)
    #     self.img_transform = img_transform
    #     self.label_transform = label_transform
    #     self.h_flip = HorizontalFlip()
    #     self.v_flip = VerticalFlip()
    #     self.test = test
    #     data_dir = root
    #
    #     import glob
    #     input_filenames = glob.glob(os.path.join(root + 'images/', '*.png'))
    #
    #     # TODO(ajaytanwani): add only 70 percent of images
    #     file_write = open(root + 'train.txt','w')
    #     for filename in input_filenames:
    #         file_write.write('%s\n' % filename)
    #
    #     file_write.close()
    #
    #     # TODO(ajaytanwani): add remaining 30 percent of images
    #     file_val = open(root + 'val.txt','w')
    #     for filename in input_filenames:
    #         file_val.write('%s\n' % filename)
    #
    #     file_val.close()
    #
    #     imgsets_dir = osp.join(data_dir, "%s.txt" % split)
    #
    #     with open(imgsets_dir) as imgset_file:
    #         for name in imgset_file:
    #             name = name.strip()
    #             img_file = osp.join(data_dir, "%s" % name)
    #             # name = name.replace('leftImg8bit','gtFine_labelTrainIds')
    #             label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
    #             self.files[split].append({
    #                 "img": img_file,
    #                 "label": label_file
    #             })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class SimRobotDataSet():
    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # Note; split "train" and "images" are SAME!!!

        # assert split in ["image_rgb", "image_depth"]

        # assert input_ch in [1, 2, 3]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        test_split_ratio = 0.2

        import glob
        image_filenames = glob.glob(os.path.join(root + 'image_rgb/', '*.png'))
        image_filenames.sort()

        label_filenames = glob.glob(os.path.join(root + 'image_labels/', '*.png'))
        label_filenames.sort()

        train_images_ids, test_images_ids = model_selection.train_test_split(np.arange(len(image_filenames)), test_size=test_split_ratio, random_state=0)

        train_images_filenames = [image_filenames[j] for j in train_images_ids]
        test_images_filenames = [image_filenames[j] for j in test_images_ids]

        train_labels_filenames = [label_filenames[j] for j in train_images_ids]
        test_labels_filenames = [label_filenames[j] for j in test_images_ids]

        file_write = open(root + 'train.txt','w')
        file_label_write = open(root + 'train_label.txt','w')
        for train_filename, label_filename in zip(train_images_filenames, train_labels_filenames):
            file_write.write('%s\n' % train_filename)
            file_label_write.write('%s\n' % label_filename)

            self.files[split].append({
                "img": train_filename,
                "label": label_filename,
            })

        file_write.close()
        file_label_write.close()

        file_write = open(root + 'val.txt','w')
        file_label_write = open(root + 'val_label.txt','w')
        for test_filename, test_label_filename in zip(test_images_filenames, test_labels_filenames):
            file_write.write('%s\n' % test_filename)
            file_label_write.write('%s\n' % test_label_filename)
        file_write.close()
        file_label_write.close()


        # file_val = open(root + 'val.txt','w')
        # for filename in test_images_filenames:
        #     file_val.write('%s\n' % filename)
        # file_val.close()
        #
        # for train_filename, label_filename in zip(train_images_filenames, train_labels_filenames):
        #     self.files[split].append({
        #         "img": train_filename,
        #         "label": label_filename,
        #     })
        #
        # with open(imgsets_dir) as imgset_file:
        #     for name in imgset_file:
        #         name = name.strip()
        #         img_file = osp.join(data_dir, "%s" % name)
        #         # name = name.replace('leftImg8bit','gtFine_labelTrainIds')
        #         label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
        #         self.files[split].append({
        #             "img": img_file,
        #             "label": label_file
        #         })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label



def get_dataset(dataset_name, split, img_transform, label_transform, test, input_ch=3):
    assert dataset_name in ["gta", "city", "test", "city16", "synthia", "sim_robot", "real_robot"]

    name2obj = {
        "gta": GTADataSet,
        "city": CityDataSet,
        "city16": CityDataSet,
        "synthia": SynthiaDataSet,
        "sim_robot": SimRobotDataSet,
        "real_robot": RealRobotDataSet,
    }
    ##Note fill in the blank below !! "gta....fill the directory over images folder.
    name2root = {
        "gta": "/home/ajaytanwani/datasets/gta_dataset_small/",  ## Fill the directory over images folder. put train.txt, val.txt in this folder
        "city": "/home/ajaytanwani/datasets/cityscape_dataset_small/",  ## ex, ./www.cityscapes-dataset.com/file-handling
        "city16": "",  ## Same as city
        "synthia": "",  ## synthia/RAND_CITYSCAPES",
        "sim_robot": "/home/ajaytanwani/datasets/sim2real_datasets/dataset_07_05_2018_2/",
        "real_robot": "/home/ajaytanwani/datasets/uncluttered+cluttered_polygon/",
    }
    dataset_obj = name2obj[dataset_name]
    root = name2root[dataset_name]

    if dataset_name == "city16":
        return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, label_type="label16")

    return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                       test=test, input_ch=input_ch)


def check_src_tgt_ok(src_dataset_name, tgt_dataset_name):
    if src_dataset_name == "synthia" and not tgt_dataset_name == "city16":
        raise AssertionError("you must use synthia-city16 pair")
    elif src_dataset_name == "city16" and not tgt_dataset_name == "synthia":
        raise AssertionError("you must use synthia-city16 pair")


def get_n_class(src_dataset_name):
    if src_dataset_name in ["synthia", "city16"]:
        return 16
    elif src_dataset_name in ["gta", "city", "test"]:
        # return 20
        return 35
    elif src_dataset_name in ["sim_robot", "real_robot"]:
        return 7
        # return 35
    else:
        raise NotImplementedError("You have to define the class of %s dataset" % src_dataset_name)


import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import resize


def get_data_iter(annotations_file, size=None, transform=None, target_transform=None, batch_size=32, num_workers=7):
    """
    读取图片数据集的label.csv文件，将会返回一个数据集的可迭代对象。
    :param annotations_file: 含有 "文件路径" 以及对应 "标签" 的csv文件名,eg：XXX.jpg,1
    :param size: 对图片做resize的操作时的size
    :param transform: 对图片做的变换
    :param target_transform: 对标签做的变换
    :param batch_size: 批大小
    :param num_workers: 处理数据的线程数
    :return: 一个数据集的可迭代对象。
    """
    img_map = _ImgLoad(annotations_file, size, transform, target_transform)
    data_iter = torch.utils.data.DataLoader(img_map, batch_size=batch_size, num_workers=num_workers)
    return data_iter


class _ImgLoad(torch.utils.data.Dataset):
    def __init__(self, annotations_file, size=None, transform=None, target_transform=None):
        """
        :param annotations_file: 含有 "文件路径" 以及对应 "标签" 的csv文件名
        :param size: 对图片做resize的操作
        :param transform: 对图片做的变换
        :param target_transform: 对标签做的变换
        """
        self.img_label = pd.read_csv(annotations_file)
        self.target_transform = target_transform
        if size:
            # 如果传入size，则忽略transform，因为resize是一个很常用的操作
            self.size = size
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = self.img_label.iloc[idx, 0]  # 获取图片文件名
        img = read_image(img_path).to(dtype=torch.float)  # 读取图片

        # 获得图片的分类
        label = self.img_label.iloc[idx, 1]

        if self.size:
            img = resize(img, size=self.size)
        elif self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)
        return img, label

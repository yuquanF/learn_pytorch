import pandas as pd
import os


def _get_imgs_path(root, suffix=None):
    """获取root目录下的所有文件路径，若suffix不为None，则只获取指定后缀的文件,如 “.jpg”
    """
    imgs_path = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if not suffix:
                imgs_path.append(os.path.join(root, file))
            elif os.path.splitext(file)[1] == suffix:  # os.path.splitext() 将文件名和扩展名分开
                imgs_path.append(os.path.join(root, file))
    return imgs_path


def save_label_to_csv():
    """获得图片的分类"""
    paths = []
    labels = []
    label_names = []
    imgs_path = _get_imgs_path('./image')

    for path in imgs_path:
        name = path.split('/')[-1]
        prefix = name.split('_')[0]
        label = 1 if prefix == 'covid' else 0
        paths.append(path)
        labels.append(label)
        label_names.append(prefix)

    df = pd.DataFrame({'paths': paths, 'label': labels, 'label_name': label_names})

    print(df[:5])

    df.to_csv('label.csv', index=False)


if __name__ == '__main__':
    save_label_to_csv()
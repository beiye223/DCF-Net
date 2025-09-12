import os
import shutil
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    # def __init__(self, data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    def __init__(self, data_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        初始化数据集划分工具类
        :param data_dir: 数据集的根目录，包含原始图像和标注
        :param output_dir: 划分后数据集保存的目录
        :param train_ratio: 训练集的比例
        :param val_ratio: 验证集的比例
        :param test_ratio: 测试集的比例
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.check_ratios()

    def check_ratios(self):
        """检查划分比例是否合理"""
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1 and 0 < self.test_ratio < 1):
            raise ValueError("划分比例必须在0到1之间")
        if self.train_ratio + self.val_ratio + self.test_ratio != 1.0:
            raise ValueError("划分比例之和必须为1")

    def create_split(self, image_path, mask_path):
        """根据比例划分数据集"""
        # 获取所有图像和标签路径
        image_paths = sorted(
            [os.path.join(self.data_dir, image_path, f) for f in os.listdir(os.path.join(self.data_dir, image_path))])
        label_paths = sorted(
            [os.path.join(self.data_dir, mask_path, f) for f in os.listdir(os.path.join(self.data_dir, mask_path))])

        # 确保图像和标签一一对应
        assert len(image_paths) == len(label_paths), "图像和标签数量不匹配"

        # 使用sklearn划分数据集
        train_images, temp_images, train_labels, temp_labels = train_test_split(image_paths, label_paths,
                                                                                test_size=1 - self.train_ratio,
                                                                                random_state=42)
        val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels,
                                                                            test_size=self.test_ratio / (
                                                                                    self.val_ratio + self.test_ratio),
                                                                            random_state=42)

        # 创建目标文件夹
        self._create_dirs()

        # 将数据拷贝到相应目录
        self._copy_files(train_images, train_labels, "train")
        self._copy_files(val_images, val_labels, "val")
        self._copy_files(test_images, test_labels, "test")

    def _create_dirs(self):
        """创建训练集、验证集和测试集的文件夹结构"""
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, "masks"), exist_ok=True)

    def _copy_files(self, images, labels, split):
        """将图像和标签拷贝到对应的文件夹"""
        for img, lbl in zip(images, labels):
            img_name = os.path.basename(img)
            lbl_name = os.path.basename(lbl)

            # 拷贝图像文件
            shutil.copy(img, os.path.join(self.output_dir, split, "images", img_name))
            # 拷贝标签文件
            shutil.copy(lbl, os.path.join(self.output_dir, split, "masks", lbl_name))

    def print_split_info(self, output_file="dataset_split_info.txt"):
        """打印划分结果的文件数量并保存到文本文件"""
        # 用于保存统计信息的文本内容
        split_info = []

        for split in ["TrainDataset", "ValidaDataset", "TestDataset"]:
            image_count = len(os.listdir(os.path.join(self.output_dir, split, "images")))
            label_count = len(os.listdir(os.path.join(self.output_dir, split, "masks")))

            # 打印到控制台
            print(f"{split.capitalize()} - Images: {image_count}, Masks: {label_count}")

            # 将统计信息保存到文本中
            split_info.append(f"{split.capitalize()} - Images: {image_count}, Masks: {label_count}")

        # 将信息写入文件
        with open(os.path.join(self.output_dir, output_file), "w") as file:
            file.write("\n".join(split_info))
        print(f"统计信息已保存到 {os.path.join(self.output_dir, output_file)}")


if __name__ == "__main__":

    data_dir = "/home/zc/thrid_passage/dataset"  # 数据集根目录路径

    image_path = "FIND/images"
    mask_path = "FIND/masks"

    dataset_name = "FIND_after"  # 划分后数据集名
    output_dir = f"/home/zc/thrid_passage/{dataset_name}"  # 保存划分后数据集的目录路径
    splitter = DatasetSplitter(data_dir, output_dir,
                               train_ratio=0.7,
                               val_ratio=0.1,
                               test_ratio=0.2)

    splitter.create_split(image_path, mask_path)
    splitter.print_split_info()

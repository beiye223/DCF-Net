import os
import shutil

# 图像和掩膜文件夹路径
images_dir = "D:\crack_segmentation\dataset\crack9000\Final-Dataset-Vol1\Final-Dataset-Vol1\Final_Masks\Heads"
masks_dir = "D:\crack_segmentation\dataset\crack9000\Final-Dataset-Vol1\Final-Dataset-Vol1\Final_Masks\Masks"

# 输出文件夹路径
output_base_dir = 'D:\crack_segmentation\裂缝分割'  # 所有类别的基础输出目录

# 创建基础输出目录（如果不存在）
os.makedirs(output_base_dir, exist_ok=True)

# 获取图像和掩膜文件的列表
image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

# 遍历图像文件并匹配对应的掩膜文件
for image_filename in image_files:
    image_path = os.path.join(images_dir, image_filename)
    print("{}".format(image_path))
    # 确保是文件而不是目录
    if os.path.isfile(image_path):
        # 假设图像文件名和掩膜文件名结构相同，只是掩膜文件名以 "_mask" 结尾
        mask_filename = image_filename.replace('.png', '_mask.png')  # 根据实际格式调整
        mask_path = os.path.join(masks_dir, mask_filename)

        # 检查是否存在对应的掩膜文件
        if os.path.exists(mask_path):
            # 从图像文件名中提取类别（这里假设类别是图像文件名前缀的一部分）
            category = image_filename.split('_')[0]  # 例如，'cat' 从 'cat_001.png' 中提取
            category_image_dir = os.path.join(output_base_dir, category, 'images')
            category_mask_dir = os.path.join(output_base_dir, category, 'masks')

            # 创建类别文件夹（如果不存在）
            os.makedirs(category_image_dir, exist_ok=True)
            os.makedirs(category_mask_dir, exist_ok=True)

            # 移动图像和掩膜到对应类别的文件夹中
            shutil.move(image_path, os.path.join(category_image_dir, image_filename))
            shutil.move(mask_path, os.path.join(category_mask_dir, mask_filename))

print("图像和掩膜分类并按类别分文件夹完成！")

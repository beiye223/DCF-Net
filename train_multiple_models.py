import time

import torch
from torch.utils.data import DataLoader
import timm

from contrast_model.CT_CrackSeg.TransMUNet import TransMUNet
from model_factory import get_model
from models.DECSNet.DECSNet import DECSNet
from models.DTrC_Net.CTC_Net import CTCNet
from models.DeeplabV3plus.DeepLabv3plus import DeepLabv3plus
from models.UKAN.kan_model import UKAN
from models.UTNet.utnet import UTNet
from models.dual_path_cracker.Net import Dual_Path_Net
from my_datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
import models.LM_Net.LM_Net as LM_Net
from models.Unet import Unet
from models.CrackFormer2 import CrackFormer2
import matplotlib.pyplot as plt
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")


def main(config, model_name):
    """


        创建日志文件和检查点文件夹


    """
    config.work_dir = f'runs/{model_name}/' + model_name + '_' + config.datasets + '_' + datetime.now().strftime(
        '%m.%d.%H.%M') + '/'
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    """


        设置GPU


    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    """


       准备数据集


    """

    train_dataset = NPY_datasets(config.data_path, config, train=True)
    print(f'========>>>>训练数据集长度为:{len(train_dataset)}')
    print(f"========>>>>训练数据集:{config.data_path}")
    print(f"========>>>>训练数据大小为:{config.input_size_h, config.input_size_w}")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    """


        配置模型


    """
    model_cfg = model_name

    model = get_model(config,model_name)
    """


        配置 loss, opt, sch and amp


    """
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    """


        Set other params
        设置其超参数


    """
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    """


      从之前保存的训练检查点（checkpoint）中恢复模型的状态，并在恢复的基础上继续训练。


    """
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    """


    模型训练


    """
    print('#----------Training----------#')
    print(f"========>>>>使用的模型为:{model_name}")

    train_loss_list = []
    val_loss_list = []

    start_time = time.time()
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step, train_epoch_loss = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch,
                                                 step, logger, config, writer)
        train_loss_list.append(train_epoch_loss)

        # val loss
        loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config)
        val_loss_list.append(loss)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    """


       测试


    """
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)

        loss, test_metrics = test_one_epoch(model_name,val_loader, model, criterion, logger, config)

        # 创建并写入TXT文件
        with open(os.path.join(config.work_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as file:
            file.write('\n>>>>>>>>>>>>')
            file.write(Modify_Description)
            file.write('<<<<<<<<<<<<<<<\n')
            file.write(test_metrics)

    # 将train_loss保存到txt文件中
    with open(os.path.join(config.work_dir, 'train_loss.txt'), 'w', encoding='utf-8') as file:
        # 遍历列表中的每个元素
        for item in train_loss_list:
            # 将元素写入文件，元素之间用逗号隔开，并在每个元素后添加换行符
            file.write(f'{item},\n')

    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    print(f"------>>>>训练用时: {hours}h {minutes}m<<<<-------------")

    epochs = range(1, len(train_loss_list) + 1)
    plt.figure()
    plt.plot(epochs, train_loss_list, 'r', label='Training loss')
    plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config.work_dir, 'loss.png'), dpi=150)
    plt.show()


if __name__ == '__main__':

    config = setting_config

    if config.train_multiple_models:
        for model in config.train_model_list:

            model_name = model

            Modify_Description = (f"dataset : {config.datasets} \n"
                                  f"")
            print(f"\n>>>>>>>>>>>>{Modify_Description}<<<<<<<<<<<<<<<")

            main(config, model_name)
    else:
        Modify_Description = (f"dataset : {config.datasets} \n"
                              f"")
        print(f"\n>>>>>>>>>>>>{Modify_Description}<<<<<<<<<<<<<<<")

        main(config)

import torch
from torch.utils.data import DataLoader
import timm
from my_datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter

from models.CMC_Net.my_model import My_Dual_Model
from models.vmunet.vmunet import VMUNet
from models.Unet import Unet
# from models.my_DualPath.my_dual_model import My_Dual_Model
from models.DTrC_Net.CTC_Net import CTCNet
from models.dual_path_cracker.Net import Dual_Path_Net
from models.DECSNet.DECSNet import DECSNet
import matplotlib.pyplot as plt
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")


def Vis_pred(msk_pred, i, pred_save_path, threshold=0.5, test_data_name=None):
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 255, 0)

    height, width = msk_pred.shape  # 获取图像的高度和宽度
    plt.figure(figsize=(width / 100, height / 100), dpi=300)  # 设置画布大小，单位为英寸
    plt.imshow(msk_pred, cmap='gray')  # 显示图像
    plt.axis('off')  # 关闭坐标轴

    if test_data_name is not None:
        save_path = pred_save_path + test_data_name + '_'
    plt.savefig(pred_save_path + str(i) + '.png',bbox_inches='tight', pad_inches=0)
    plt.close()


def main(config, dataset_path, test_Model_weight_path, pred_save_path):
    # print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    # print('#----------Preparing dataset----------#')
    test_dataset = NPY_datasets(dataset_path, config, train=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=False)

    # print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    print(f'==============Test Model : {config.network}==============')

    if config.network == 'vmunet':
        model = VMUNet(num_classes=model_cfg['num_classes'], input_channels=model_cfg['input_channels'], depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'], drop_path_rate=model_cfg['drop_path_rate'],load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()


    elif config.network == 'Unet':

        model = Unet()

    elif config.network == 'My_Dual_Model':

        model = My_Dual_Model(in_channels=3, out_channels=1)

    elif config.network == 'DTrC_Net':

        model = CTCNet(n_channels=3, n_classes=1, img_size=config.input_size_h)

    elif config.network == 'Dual_Path_Net':

        model = Dual_Path_Net()

    elif config.network == 'DECSNet':

        model = DECSNet(3, 64, 512, 4, 'resnet50',

                        False, [256, 512, 1024, 2048], [64, 128, 256, 512],

                        1).to('cuda')
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    # print('#----------Prepareing loss, opt, sch and amp----------#')

    if os.path.exists(test_Model_weight_path):
        print('#----------Testing----------#')
        best_weight = torch.load(test_Model_weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight, strict=False)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                img, msk = data
                img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

                out = model(img)

                if type(out) is tuple:
                    out = out[0]
                out = out.squeeze(1).cpu().detach().numpy()

                Vis_pred(out,i, pred_save_path)


if __name__ == '__main__':
    config = setting_config
    test_Model_weight_path = 'runs/AS300/checkpoints/best.pth'
    dataset_path = '/home/zc/消融图示/Dataset/'
    pred_save_path = '/home/zc/消融图示/Vis_Pred/'
    main(config, dataset_path, test_Model_weight_path, pred_save_path)

import torch

# from models.PSP_Net import pspnet
# from models.Unet_plus_plus import UnetPlusPlus
# from models_new_6_4.AFENet.AFENet import AFENet
# from models.CT_CrackSeg.TransMUNet import TransMUNet
# from models.DECSNet.DECSNet import DECSNet
# from models.DTrC_Net.CTC_Net import DTrC_Net
# from models.RS3mamba.RS3mamba import RS3Mamba
# from models.SFFNet.SFFNet import SFFNet
# from models.UKAN.kan_model import UKAN
# from models.UTNet.utnet import UTNet
# from models.CMC_Net.my_model import CMC_Net
# from models.dual_path_cracker.Net import Dual_Path_Net
# from models.LM_Net.LM_Net import LM_Net
# from models.CrackFormer2 import CrackFormer2
from models.Unet import Unet
# from models_new_6_4.DcsNet import DcsNet
# from models_new_6_4.DeepCrack import DeepCrack
# from models_new_6_4.JTFN_main import JTFN
# from models_new_6_4.SegNet import SegNet
# from models_new_6_4.U_Net_like import U_Net_like
# from models_new_6_4.V2Net_main.v2net import V2net

from MyBackBone.thrid_kuangjia.upper_net import UPerNet


def get_model(config,model_name):
    """
    根据config中的网络名称返回相应的模型。
    :param config: 配置文件对象，包含模型网络的配置信息
    :return: 选定的模型实例
    """
    params = {'in_chns': 3,
              # 'feature_chns': [32, 64, 128, 256, 512],
              'feature_chns': [16, 32, 64, 128, 256],
              'dropout': [0.05, 0.05, 0.05, 0.05, 0],
              'class_num': 1,
              'up_type': 1,
              'acti_func': 'relu'}

    if model_name == 'Unet':
        return Unet().cuda()
    elif model_name == 'uppernet':
        return UPerNet(num_classes=1, params1=params).cuda()
    elif model_name == 'AFENet':
        return AFENet().to('cuda')
    elif model_name =='JTFN':
        return JTFN().to('cuda')
    elif model_name =='V2net':
        return V2net().to('cuda')
    elif model_name == "DeepCrack_Net":
        return DeepCrack().cuda()
    elif model_name == 'SegNet':
        return SegNet().to('cuda')
    elif model_name =='U_Net_like':
        return U_Net_like().cuda()
    elif model_name =='PSP_Net':
        return pspnet().cuda()
    elif model_name == 'UnetPlusPlus':
        return UnetPlusPlus().cuda()
    elif model_name == 'UKAN':
        return UKAN(num_classes=1, img_size=config.input_size_h, embed_dims=[128, 256, 512]).cuda()
    elif model_name == 'UTNet':
        return UTNet(in_chan=3, base_chan=48, num_classes=1).cuda()
    elif model_name == 'CMC_Net':
        return CrackFormer2.crackformer2().cuda()
    elif model_name == 'CT_CrackSeg':
        return TransMUNet(n_classes=1).cuda()
    elif model_name == 'SFFNet':
        return SFFNet(num_classes=1).cuda()
    elif model_name == 'RS3Mamba':
        return RS3Mamba().cuda()
    elif model_name == 'DECSNet':
        return DECSNet(3, 64, 512, 4, 'resnet50', False,
                       [256, 512, 1024, 2048], [64, 128, 256, 512], 1).to('cuda')
    elif model_name == 'Dual_Path':
        return Dual_Path_Net().cuda()
    elif model_name == 'DcsNet':
        return DcsNet().cuda()


    elif model_name == 'DTrC_Net':
        return DTrC_Net(n_channels=3, n_classes=1, img_size=config.input_size_h).cuda()
    elif model_name == 'LM_Net':
        return LM_Net().cuda()

    else:
        raise ValueError(f"模型未定义|| Unknown network: {model_name}")

if __name__ == '__main__':


    model = get_model(config='',model_name='uppernet')
    input = torch.randn(4, 3, 512, 512).to("cuda")
    out = model(input)

    print("输入维度：",input.size())
    print("output维度：",out.size())
    print('#Parameters:', sum(param.numel() for param in model.parameters()))

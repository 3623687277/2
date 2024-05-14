import numpy as np
import torch
import torch.nn as nn
###修改的地方
from .attention_2d import PSAModule
from .CSPdarknet import C3, Conv, CSPDarknet, Bottleneck, SiLU
import matplotlib.pyplot as plt


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        num_levels = len(layer_nums)
        #函数在调用多个参数时，在列表、元组、集合、字典及其他可迭代对象作为实参，并在前面加 *
        #如*（1,2,3）解释器将自动进行解包然后传递给多个单变量参数（参数个数要对应相等）
        #num_filters:[64,128,256];c_in_list=[input_channels,64,128,256]
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        # -----------------------------------------------#
        # -----------------------------------------------#
        #增加的部分
        self.attentions = nn.ModuleList()
        # -----------------------------------------------#
        # -----------------------------------------------#
        # 修改下采样部分，增加残差结构，增强特征提取
        for idx in range(num_levels):
            # 利用卷积进行高和宽的压缩，并扩张通道数
            cur_layers = [
                #最常用的零填充函数是nn.ZeroPad2d，也就是对Tensor使用0进行边界填充，我们可以指定tensor的四个方向上的填充数
                #nn.ZeroPad2d(padding=(1, 2, 3, 4))
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                # 利用C3结构进行残差模块的构建
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),

                    # -----------------------------------------------#
                    # -----------------------------------------------#
                    # 添加残差模块
                    C3(num_filters[idx], num_filters[idx])
                    # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    # SiLU()
                    # nn.ReLU()

                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            # -----------------------------------------------#
            # -----------------------------------------------#
            #增加的部分
            self.attentions.append(nn.Sequential(
                PSAModule(num_filters[idx], num_filters[idx], stride=1, conv_kernels=[3, 5, 7, 9],
                          conv_groups=[1, 4, 8, 16]),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                SiLU()
                # nn.ReLU()
            ))
            # -----------------------------------------------#
            # -----------------------------------------------#



            # 上采样部分要保留
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        SiLU()
                        # nn.ReLU()
                    ))
                else:
                    #对浮点数取整
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        # nn.ReLU()
                        SiLU()
                    ))
        #python自带的sum函数（或者Numpy中的sum函数），无参时，所有全加；axis=0，按列相加；axis=1，按行相加；
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                #nn.ConvTranspose2d的功能是进行反卷积操作
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                SiLU(),
                # nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features    ##x[1,64,496,432]
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # print('*' * 100)
            # exit()
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            #增加的部分
            x = self.attentions[i](x)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict



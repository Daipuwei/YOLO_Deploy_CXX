# -*- coding: utf-8 -*-
# @Time    : 2024/4/18 9:00
# @Author  : DaiPuWei
# @File    : model2header.py
# @Software: PyCharm

"""
    这是将模型权重文件及其相关参数转换为头文件的脚本
"""

import os
import io
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate binary header output')
parser.add_argument('--model_path', type=str, default="",help='model path ')
parser.add_argument('--label_txt_path', type=str,default="",help='class name txt path')
parser.add_argument('--stride_txt_path', type=str,default="",help='stride txt path')
parser.add_argument('--anchor_txt_path', type=str,default="",help='anchor txt path')
parser.add_argument('--header_path',type=str,default="",help='output header path')
parser.add_argument('--project_name', type=str,default="",help='project name')
opt = parser.parse_args()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def encrypt(data, adder=3):
    data_len = len(data)
    a = random.sample(range(data_len), 4)
    for i in a:
        new_i = data[i] + adder
        if new_i <= 255:
            data[i] = new_i

def data2header(data_dict, header_path, project_name):
    """
    这是相关数据转换为头文件的函数
    Args:
        data_dict: 数据字典
        header_path: 头文件路径
        project_name: 项目名称
    Returns:
    """
    # 初始化头文件名称
    _,header_name = os.path.split(header_path)
    fname,ext = os.path.splitext(header_name)

    # 初始化头文件内容文本数组
    out = []
    out.append("#ifndef {0}_{1}_H".format(project_name.upper(),fname.upper()))
    out.append("#define {0}_{1}_H".format(project_name.upper(),fname.upper()))
    out.append("#include <vector>")
    out.append("#include <string>")
    out.append("#include \"stdint.h\"")

    # 将模型权重转换为二进制数组
    out.append('unsigned char {0}_0[] = {{'.format(project_name))
    model_weight_array = data_dict["model_weight"]
    l = [ model_weight_array[i:i+12] for i in np.arange(0, len(model_weight_array), 12) ]
    for i in tqdm(np.arange(len(l))):
        line = ', '.join([ '0x{0:02x}'.format(c) for c in l[i]])
        out.append('  {0}{1}'.format(line,',' if i<len(l)-1 else ''))
    out.append('};')
    out.append('unsigned int {0}_0_len = {1};'.format(project_name, len(model_weight_array)))

    # 合并所有模型权重及其长度到数组中
    data_arr_names = ["(const void *){0}_0".format(project_name)]
    data_len_names = ["{0}_0_len".format(project_name)]
    out.append("std::vector<const void *> {0}s = ".format(project_name) +"{" + ','.join(data_arr_names) + "};")
    out.append("std::vector<unsigned int> {0}_lens = ".format(project_name) + "{" + ','.join(data_len_names) + "};")

    # 初始化标签名称数组
    label_names = data_dict["label_names"]
    if len(label_names) > 0:
        out.append("std::vector<std::string> {0}_label_data = ".format(project_name)+"{")
        l = [ label_names[i:i+12] for i in np.arange(0, len(label_names), 12) ]
        for i in tqdm(np.arange(len(l))):
            line = '\"'+'", "'.join(l[i])+'\"';
            out.append('  {0}{1}'.format(line,',' if i<len(l)-1 else ''))
        out.append('};')

    # 初始化下采样率数组
    strides = data_dict["strides"]
    if len(strides) > 0:
        line = "std::vector<int> {0}_strides = ".format(project_name)+"{"
        line += ','.join([str(stride) for stride in strides])
        line += "};"
        out.append(line)

    # 初始化anchor尺度数组
    anchors = data_dict["anchors"]
    if len(anchors) > 0:
        out.append("std::vector<std::vector<float>> {0}_anchors = ".format(project_name)+"{")
        for _anchors in anchors:
            line = "  {"+",".join([str(anchor) for anchor in _anchors])+"}"
            out.append('  {0}{1}'.format(line,',' if i<len(anchors)-1 else ''))
        out.append('};')

    #  初始化头文件结尾
    out.append("#endif // {0}_{1}_H".format(project_name.upper(),fname.upper()))

    # 写入头文件
    out = '\n'.join(out)
    with open(header_path, 'w', encoding='utf-8') as f:
        f.write(out)

def model2header(model_path,header_path,project_name,label_txt_path=None,stride_txt_path=None,anchor_txt_path=None):
    """
    这是将模型权重转换为头文件的函数
    Args:
        model_path: 模型权重文件路径
        header_path: 头文件路径
        project_name: 项目名称
        label_txt_path: 标签名称txt文件路径, 默认为None
        stride_txt_path: 下采样率txt文件路径, 默认为None
        anchor_txt_path: anchor尺度txt文件路径，默认为None
    Returns:
    """
    # 初始化数据列表
    data_dict = {}
    # 读取模型权重文件
    with open(model_path, 'rb') as f:
        model_data = bytearray(f.read())
        data_dict["model_weight"] = model_data

    # 读取标签文件
    label_names = []
    if label_txt_path is not None:
        with open(label_txt_path, encoding='utf8') as f:
            for line in f.readlines():
                label_names.append(line.strip())
    data_dict["label_names"] = label_names

    # 读取下采样数组
    strides = []
    if stride_txt_path is not None:
        with open(stride_txt_path, encoding='utf8') as f:
            for line in f.readlines():
                strides.append(int(line.strip()))
    data_dict["strides"] = strides

    # 读取anchor尺寸数组
    anchors = []
    if anchor_txt_path is not None:
        with open(anchor_txt_path, encoding='utf8') as f:
            for line in f.readlines():
                anchors.append(float(line.strip()))
        anchors = np.array(anchors)
        anchors = np.reshape(anchors,(len(strides),-1))
        anchors = anchors.tolist()
    data_dict['anchors'] = anchors

    # 将权重文件及其相关参数转换为头文件
    data2header(data_dict,header_path,project_name)

def run_main():
    """
    这是主函数
    Returns:
    """
    # 解析参数
    model_path = opt.model_path
    header_path = opt.header_path
    project_name = opt.project_name
    label_txt_path = opt.label_txt_path
    stride_txt_path = opt.stride_txt_path
    anchor_txt_path = opt.anchor_txt_path
    if label_txt_path == "":
        label_txt_path = None
    if stride_txt_path == "":
        stride_txt_path = None
    if anchor_txt_path == "":
        anchor_txt_path = None
    model2header(model_path,header_path,project_name,label_txt_path,stride_txt_path,anchor_txt_path)

if __name__ == '__main__':
    run_main()

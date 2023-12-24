# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os

import numpy as np
import scanpy as sc

import logger as l
import load_data
import st_acn
import pickle
import cluster
import pandas as pd

# 配置日志记录
from src import enhence
def parse_args():
    # common parser
    parser = argparse.ArgumentParser(description='example')

    # 添加位置参数
    parser.add_argument('input_data_path', type=str, help='path of input data')
    parser.add_argument('section_id', type=str, help="section_id")
    parser.add_argument('output_data_path', type=str, help='path of input data')

    # 添加可选参数
    parser.add_argument('-v', '--value', type=str, help="value", default="value1")

    # 解析命令行参数
    args = parser.parse_args()
    print(f'run with args = {args}')
    return args

# 算法核心逻辑
def call(low_dim_x,cell_spatial,ground_truth):
    z_all_dump = './z_all.pkl'
    # if os.path.exists(z_all_dump):
    #     with open(z_all_dump, 'rb') as z_all_dump_file:
    #         Z_all = pickle.load(z_all_dump_file)
    # else:
    #     Z_all = st_acn.stACN(low_dim_x, cell_spatial.A, ground_truth)
    #     with open(z_all_dump, 'wb') as z_all_dump_file:
    #         pickle.dump(Z_all, z_all_dump_file)
    l.logger.info(f'[call] run st_acn_master')
    Z_all = st_acn.stACN(low_dim_x, cell_spatial.A, ground_truth)
    with open(z_all_dump, 'wb') as z_all_dump_file:
        pickle.dump(Z_all, z_all_dump_file)
    l.logger.info(f'[call] run st_acn_master done')
    cluster.cluster(Z_all,ground_truth)
    pass



def process_data(input_data_path:str, section_id:str):
    l.logger.info(f'[process_data] begin ')
    low_dim_x_dump = "./low_dim_x.pkl"
    enhanced_adata_dump = './enhanced_adata.pkl'
    cell_spatial_dump = './cell_spatial.pkl'
    ground_truth_dump = './ground_truth.pkl'
    if os.path.exists(low_dim_x_dump) and\
            os.path.exists(enhanced_adata_dump) and\
            os.path.exists(cell_spatial_dump) and\
            os.path.exists(ground_truth_dump) :
        l.logger.info(f'[LoadPkl] loading ...')
        with open(low_dim_x_dump,"rb") as low_dim_x_file:
            low_dim_x = pickle.load(low_dim_x_file)
        with open(enhanced_adata_dump, "rb") as enhanced_adata_file:
            enhanced_adata = pickle.load(enhanced_adata_file)
        with open(cell_spatial_dump, "rb") as cell_spatial_file:
            cell_spatial = pickle.load(cell_spatial_file)
        with open(ground_truth_dump, "rb") as ground_truth_file:
            ground_truth = pickle.load(ground_truth_file)
        l.logger.info(f'[LoadPkl]  done')
    else:
        l.logger.info(f'[LoadData] loading ...')

        #  加载数据
        AnnData = load_data.load_data_for_h5(input_data_path, section_id)
        print(AnnData)

        # enhence data
        l.logger.info(f'[enhence_data] begin')
        enhanced_adata ,cell_spatial = enhence.enhence_data(AnnData)
        l.logger.info(f'[enhence_data] end')

        # load truth
        ground_truth = load_data.load_data_for_groud_truth(input_data_path, section_id, AnnData)
        low_dim_x = enhanced_adata.obsm['X_pca']

        l.logger.info(f'[DumpPkl] loading ...')
        with open(low_dim_x_dump, "wb") as low_dim_x_file:
            pickle.dump(low_dim_x,low_dim_x_file)
        with open(enhanced_adata_dump, "wb") as enhanced_adata_file:
            pickle.dump(enhanced_adata,enhanced_adata_file)
        with open(cell_spatial_dump, "wb") as cell_spatial_file:
            pickle.dump(cell_spatial, cell_spatial_file)
        with open(ground_truth_dump, "wb") as ground_truth_file:
            pickle.dump(ground_truth,ground_truth_file)
        l.logger.info(f'[DumpPkl] done')
    l.logger.info(f'[process_data] done ')
    return low_dim_x,enhanced_adata,cell_spatial,ground_truth

# main func
def main():
    # 解析命令行参数
    args = parse_args()

    # 读取参数
    input_data_path = args.input_data_path
    section_id = args.section_id
    output_data_path = args.output_data_path
    log_path = os.path.join(output_data_path,'log')

    # 初始化日志
    l.initlog(log_path)

    # 打印启动日志
    l.logger.info(f'main args={args}')

    # 准备数据
    low_dim_x,enhanced_adata,cell_spatial,ground_truth = \
        process_data(input_data_path,section_id)

    # call
    call(low_dim_x,cell_spatial,ground_truth)

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
RNA 二级结构生成工具 - 纯 Python API 版本

此脚本使用 ViennaRNA 的 Python API 处理 FASTA 文件中的多个 RNA 序列，
并生成不显示碱基的二级结构图。

用法：
    python RNA_SS.py 输入文件.fasta 输出目录
"""

import os
import sys
import RNA
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def fold_rna(seq):
    """使用 ViennaRNA 折叠 RNA 序列"""
    # 正确调用 RNA.fold 函数 - 不传入 md 参数
    (ss, mfe) = RNA.fold(seq)
    print(f"最小自由能: {mfe:.2f} kcal/mol")
    
    # 获取碱基对列表
    pt = RNA.ptable(ss)
    pairs = []
    for i in range(1, len(pt)):
        if pt[i] > i:  # 只考虑前向碱基对
            pairs.append((i-1, pt[i]-1))  # 调整为从0开始的索引
    
    # 获取二级结构的坐标
    coords = RNA.get_xy_coordinates(seq, ss)
    
    return ss, mfe, pairs, coords

def plot_structure(seq, ss, coords, pairs, output_file):
    """绘制没有碱基标签的 RNA 二级结构图"""
    # 创建新图形
    seq_len = len(seq)
    fig_size = max(8, seq_len / 30)  # 根据序列长度调整图形大小
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # 提取坐标
    x = []
    y = []
    for i in range(len(seq)):
        x.append(coords.get(i).X)
        y.append(coords.get(i).Y)
    
    # 绘制骨架
    for i in range(1, len(seq)):
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]], 'k-', linewidth=1.5, alpha=0.7)
    
    # 绘制碱基对
    for i, j in pairs:
        ax.plot([x[i], x[j]], [y[i], y[j]], '-', color='blue', linewidth=1.0, alpha=0.6)
    
    # 绘制节点位置（不显示碱基）
    for i in range(len(seq)):
        circle = Circle((x[i], y[i]), 0.2, facecolor='lightgray', edgecolor='gray')
        ax.add_patch(circle)
    
    # 设置等比例和隐藏坐标轴
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 增加一些边距
    plt.tight_layout(pad=1.0)
    
    # 保存图形
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结构图已保存到 {output_file}")

def process_sequence(seq, seq_name, output_dir):
    """处理单个 RNA 序列并生成结构图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 折叠 RNA
    ss, mfe, pairs, coords = fold_rna(seq)
    
    # 生成输出文件路径
    output_file = os.path.join(output_dir, f"{seq_name}.pdf")
    
    # 绘制结构图
    plot_structure(seq, ss, coords, pairs, output_file)
    
    return True

def process_fasta(input_file, output_dir):
    """处理 FASTA 文件中的所有序列"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个序列
    for record in SeqIO.parse(input_file, "fasta"):
        # 清理序列名用于文件名
        seq_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in record.id)
        seq = str(record.seq).upper().replace('T', 'U')  # 将 T 转换为 U
        
        print(f"处理序列: {seq_name}")
        process_sequence(seq, seq_name, output_dir)
        print(f"成功处理 {seq_name} ({len(seq)} nt)")

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <输入FASTA文件> <输出目录>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"未找到输入文件: {input_file}")
        sys.exit(1)
    
    # 处理 FASTA 文件
    print(f"开始处理文件 {input_file}")
    process_fasta(input_file, output_dir)
    print(f"完成: 所有 RNA 结构图已保存到 {output_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FASTA 文件拆分工具
这个脚本将一个包含多个序列的 FASTA 文件拆分成多个单序列的 FASTA 文件，
每个文件以序列的 ID 命名。所有序列将被转换为单行格式。
用法：
    python split_fasta.py input.fasta [output_directory]
"""
import os
import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def split_fasta(input_file, output_dir=None):
    """将多序列 FASTA 文件拆分为多个单序列的 FASTA 文件，且序列强制为单行显示"""
    if output_dir is None: output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    for record in SeqIO.parse(input_file, "fasta"):
        # 获取序列 ID
        seq_id = record.id
        # 清理 ID 作为文件名
        clean_id = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in seq_id)
        
        # 创建输出文件路径
        output_file = os.path.join(output_dir, f"{clean_id}.fasta")
        
        # 手动写入，确保序列在单行上
        with open(output_file, "w") as out_handle:
            # 写入头部行
            out_handle.write(f">{record.description}\n")
            # 写入序列，确保在单行上（不使用SeqIO.write）
            out_handle.write(f"{str(record.seq)}\n")
        
        count += 1
        print(f"已保存: {output_file}")
    
    print(f"完成! 总共拆分了 {count} 个序列。")

def main():
    """主函数"""
    if len(sys.argv) < 2: 
        print(f"用法: {sys.argv[0]} <输入FASTA文件> [输出目录]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = None if len(sys.argv) <= 2 else sys.argv[2]
    
    if not os.path.isfile(input_file): 
        print(f"错误: 输入文件 '{input_file}' 不存在")
        sys.exit(1)
    
    # 拆分 FASTA 文件
    split_fasta(input_file, output_dir)

if __name__ == "__main__": 
    main()
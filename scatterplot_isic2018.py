import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def create_network_performance_plot(csv_file_path, output_path='ISIC2017_miou.png'):
    """
    创建网络性能可视化图表
    
    参数:
    csv_file_path: CSV文件路径
    output_path: 输出图片路径
    """
    
    # 读取CSV文件，尝试不同的编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
            print(f"成功读取数据，使用编码: {encoding}，共{len(df)}行")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用编码 {encoding} 读取失败: {e}")
            continue
    
    if df is None:
        print("尝试了所有常见编码格式，都无法读取CSV文件")
        print("请检查文件格式或手动指定编码格式")
        return
    
    # 检查必要的列是否存在
    required_columns = ['Network', 'mIoU(%)', 'Params(M)', 'GFLOPs']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必要的列: {missing_columns}")
        print(f"现有列: {list(df.columns)}")
        return
    
    # 数据预处理
    df = df.dropna(subset=required_columns)  # 删除缺失值
    
    # 设置图表样式
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建颜色映射 (从浅黄色到深橙色，类似原图)
    colors = ['#FFFF99', '#FFE55C', '#FFCC00', '#FF9900', '#FF6600', '#CC3300']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # 标准化GFLOPs值用于颜色映射
    gflops_normalized = (df['GFLOPs'] - df['GFLOPs'].min()) / (df['GFLOPs'].max() - df['GFLOPs'].min())
    
    # 创建散点图
    scatter = ax.scatter(df['Params(M)'], df['mIoU(%)'], 
                        c=df['GFLOPs'], cmap=cmap, 
                        s=800, alpha=0.8, edgecolors='none')
    
    # 添加网络名称标注
    for idx, row in df.iterrows():
        network_name = row['Network']
        print(f"处理网络: {network_name}")  # 添加调试信息
        # 根据网络名称设置不同的标注位置
        if network_name == 'LB-Unet':
            # 放到点的左侧
            offset_x = 12.5
            offset_y = -15
            ha = 'right'
            va = 'center'
            print("应用U-Net V2特殊位置")
        elif network_name == 'TransFuse':
            offset_x = 12.5
            offset_y = -15
            ha = 'right'
            va = 'center'
        else:
            # 默认位置：右上方
            offset_x = 12.5
            offset_y = 9.1
            ha = 'left'
            va = 'bottom'
        
        # 如果是"SELUNet(Ours)"，使用红色字体，否则使用黑色
        text_color = 'red' if row['Network'] == 'SELUNet(Ours)' else 'black'
        
        ax.annotate(row['Network'], 
                   (row['Params(M)'], row['mIoU(%)']),
                   xytext=(offset_x, offset_y), 
                   textcoords='offset points',
                   fontfamily='Times New Roman',
                   fontsize=10,
                   fontweight='bold',
                   color=text_color,
                   ha='left',
                   va='bottom')
    
    # 设置坐标轴
    ax.set_xlabel('Params (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mIoU', fontsize=12, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置坐标轴范围，留出边距
    x_margin = (df['Params(M)'].max() - df['Params(M)'].min()) * 0.1
    y_margin = (df['mIoU(%)'].max() - df['mIoU(%)'].min()) * 0.05
    
    ax.set_xlim(df['Params(M)'].min() - x_margin, df['Params(M)'].max() + x_margin)
    ax.set_ylim(df['mIoU(%)'].min() - y_margin, df['mIoU(%)'].max() + y_margin)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('GFLOPs', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 调整颜色条的高度使其与纵轴对齐
    cbar_pos = cbar.ax.get_position()
    ax_pos = ax.get_position()
    cbar.ax.set_position([cbar_pos.x0, ax_pos.y0, cbar_pos.width, ax_pos.height])
    
    # 设置图表标题
    # plt.title('Network Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()

def create_sample_data():
    """
    创建示例数据文件
    """
    sample_data = {
        'Network': ['EGE-UNet', 'MALUNet', 'MobileViTv2', 'UNeXt-S', 'MobileNetv3', 
                   'UTNetV2', 'UNet', 'TransFuse'],
        'mIoU(%)': [79.6, 78.8, 78.7, 78.3, 77.7, 77.3, 77.0, 79.2],
        'Params(M)': [2.1, 1.8, 2.3, 1.5, 3.2, 12.8, 7.8, 26.1],
        'GFLOPs': [1.2, 2.1, 3.4, 2.8, 4.2, 8.5, 6.7, 12.3]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_network_data.csv', index=False)
    print("示例数据文件 'sample_network_data.csv' 已创建")
    return df

# 使用示例
if __name__ == "__main__":
    # 创建示例数据（如果你还没有CSV文件）
    # sample_df = create_sample_data()
    # print("示例数据:")
    # print(sample_df)
    # print("\n" + "="*50 + "\n")
    
    # 使用示例数据创建图表
    # create_network_performance_plot('C:/Users/45730/Desktop/isic17.csv')
    
    # 如果你有自己的CSV文件，使用以下代码：
    create_network_performance_plot('C:/Users/45730/Desktop/ISIC2018.csv', 'ISIC2018_miou.png')

"""
我需要实现这样一个功能，请你给我用python实现： 
1. 给你一个表格，文件类型为csv
2. 有“Network”“mIoU(%)”“Params(M)“GFLOPs“ 这四列
3. 以“mIoU(%)”为纵坐标轴，“Params(M)“为横坐标轴，在图表中标注对应的“Network”。一个“Network”对应图中一个点，每个点旁边用黑色加粗的times new roman字体标注”Network“名称字体大小可以调整
4. 不同的点颜色不同，颜色越深表示该“Network”对应的“GFLOPs“值越大，颜色越浅表示”Network“对应的“GFLOPs“值越小
5. 在图标旁边加一个矩形颜色框，高度和纵轴长度保持一致。表示”Network“对应的点的颜色的渐变。给读者以参考。颜色可以设置
6.最后生成的图像样式参考这幅图片
"""
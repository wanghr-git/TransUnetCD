import matplotlib.pyplot as plt
import json


def plot(path="SF_ResNetlog.log"):
    with open(path, 'r', encoding='UTF-8') as f:
        log = json.load(f)

    y_axis_data_1 = log['train_loss']
    y_axis_data_2 = log['valid_loss']
    y_axis_data_3 = log['train_acc']
    y_axis_data_4 = log['valid_acc']
    x = range(len(log['train_loss']))

    plt.figure(figsize=(8, 4))
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.subplot(2, 2, 1)
    plt.plot(x, y_axis_data_1, '-', color='r', alpha=0.8, linewidth=1, label='train_loss')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('train_loss')

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.subplot(2, 2, 2)
    plt.plot(x, y_axis_data_2, '-', color='r', alpha=0.8, linewidth=1, label='valid_loss')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('valid_loss')

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.subplot(2, 2, 3)
    plt.plot(x, y_axis_data_3, '-', color='#4169E1', alpha=0.8, linewidth=1, label='train_acc')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('train_acc')

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.subplot(2, 2, 4)
    plt.plot(x, y_axis_data_4, '-', color='#4169E1', alpha=0.8, linewidth=1, label='valid_acc')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('valid_acc')

    plt.show()
    plt.savefig('.jpg')  # 保存该图片


if __name__ == '__main__':
    plot()

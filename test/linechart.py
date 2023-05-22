from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

def data_get(filename, dataname):
    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()
    val = ea.scalars.Items(dataname)
    val_x, val_y = [], []
    for item in val:
        val_x.append(item.step)
        val_y.append(round(item.value, 2))
    return val_x, val_y

if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
    # sns.palplot(sns.color_palette("hls", 8))
    # 加载日志数据
    x1, y1 = data_get('./datadir/69.80_Adapter_ResNet_56_Cifar100_events.out.tfevents', 'accuracy/test accuracy')
    x2, y2 = data_get('./datadir/70.28_Adapter_ResNet_56_Cifar100_events.out.tfevents', 'accuracy/test accuracy')
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    # print(y1)
    # print(y2)
    plt.ylim(65, 75)
    plt.xlim(250, 300)
    plt.savefig('saved_images/image3'+ '.png', dpi=600, pad_inches=0.3, bbox_inches="tight")
    plt.show()

    x3, y3 = data_get('./datadir/71.85_Adapter_ResNet_56_Cifar100_events.out.tfevents', 'accuracy/test accuracy')
    x4, y4 = data_get('./datadir/72.29_Adapter_ResNet_56_Cifar100_events.out.tfevents', 'accuracy/test accuracy')
    plt.plot(x3, y3)
    plt.plot(x4, y4)
    print(y3)
    print(y4)
    plt.ylim(65, 75)
    plt.xlim(250, 300)
    plt.savefig('saved_images/image4'+ '.png', dpi=600, pad_inches=0.3, bbox_inches="tight")
    plt.show()
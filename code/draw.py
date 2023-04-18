import matplotlib.pyplot as plt


def get_list(path):
    result = []
    with open(path, 'r') as f:
        for line in f:
            result.append(float(line.split(' ')[-1]))
    return result


if __name__ == '__main__':
    result_5 = get_list('../logs/0413_5_filewriting.log')
    result_10 = get_list('../logs/0413_10_filewriting.log')
    result_15 = get_list('../logs/0413_15_filewriting.log')
    result_16 = get_list('../logs/0413_16_filewriting.log')
    result_17 = get_list('../logs/0413_17_filewriting.log')
    result_18 = get_list('../logs/0413_18_filewriting.log')
    result_19 = get_list('../logs/0413_19_filewriting.log')
    result_20 = get_list('../logs/0413_20_filewriting.log')
    stage3 = get_list('../logs/stage3.log')

    x = range(1, 21, 1)
    # plt.plot(x, result_5, 'o-', label='5 clients')
    # plt.plot(x, result_10, 'o-', label='10 clients')
    # plt.plot(x, result_15, 'o-', label='15 clients')
    # plt.plot(x, result_16, 'o-', label='16 clients')
    # plt.plot(x, result_17,'o-', label='17 clients')
    # plt.plot(x, result_18,'o-', label='18 clients')
    # plt.plot(x, result_19,'o-', label='19 clients')
    plt.plot(x, stage3,'o-', label='stage3')
    plt.plot(x, result_20[:20],'o-', label='stage1')

    plt.title('Socket update and fileread update')
    plt.xlabel('Training epoch')
    plt.ylabel('Precision')

    # 添加图例
    plt.legend()
    plt.savefig('result2.png')
    # 显示图形
    plt.show()

# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/1/28 2:18
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_save_Result(final_acc_list, model_name, dataset='Benchmark', UD=0, ratio=1, win_size='1', text=True):
    '''
    :param final_acc_list: The final accuracy list
    :param model_name: The name of model to be validated
    :param dataset: The name of dataset to be validated
    :param UD: -1——Unsupervised, 0——User-Dependent；1——User-Independent
    :param ratio: -1——Training-Free, 1——80% vs 20%;2——50% vs 50%;3——20% vs 80%(UD Approach)
                  0 or else——(N-1)/N vs 1/N(UI Approach)
    :param win_size: The window size used in data to evaluate model performance
    :param text: Is the specific value displayed on the histogram
    :return:
           Show the result figure and save it
    '''
    if ratio == -1:
        proportion = 'Training-Free'
    elif ratio == 1:
        proportion = '8vs2'
    elif ratio == 2:
        proportion = '5vs5'
    elif ratio == 3:
        proportion = '2vs8'
    else:
        proportion = 'N-1vs1'

    if UD == -1:
        val_way = 'Unsupervised'
    elif UD == 0:
        val_way = 'PerSubject'
    else:
        val_way = 'CrossSubject'

    final_acc_list = np.asarray(final_acc_list)
    final_mean_list = np.mean(final_acc_list, axis=0)
    final_var_list = np.std(final_acc_list, ddof=1, axis=0)  # ddof: default——divide N(biased);1——divide N-1(unbiased)
    final_mean_list = np.append(final_mean_list, np.mean(final_mean_list, axis=0))
    final_var_list = np.append(final_var_list, np.std(final_mean_list, ddof=1, axis=0))
    print("final_mean_list.shape:", final_mean_list.shape)
    print("final_var_list.shape:", final_var_list.shape)

    # ==== 新增ITR计算 ====
    from etc.global_config import config
    def calc_itr(acc, n_class, trial_time):
        if acc <= 0 or n_class <= 1:
            return 0.0
        if acc >= 1:
            acc = 0.9999
        term1 = np.log2(n_class)
        term2 = acc * np.log2(acc) if acc > 0 else 0
        term3 = (1 - acc) * np.log2((1 - acc) / (n_class - 1)) if acc < 1 else 0
        itr = (term1 + term2 + term3) * 60 / trial_time
        return itr
    Nf = config["data_param"]["Nf"]
    ws = config["data_param"]["ws"]

    # 计算每个Fold和Mean的ITR
    itr_matrix = np.zeros_like(final_acc_list)
    for i in range(final_acc_list.shape[0]):
        for j in range(final_acc_list.shape[1]):
            itr_matrix[i, j] = calc_itr(final_acc_list[i, j], Nf, ws)
    mean_acc = np.mean(final_acc_list, axis=0)
    mean_itr = np.mean(itr_matrix, axis=0)
    std_acc = np.std(final_acc_list, ddof=1, axis=0)
    std_itr = np.std(itr_matrix, ddof=1, axis=0)

    # 增加Mean±Std
    mean_acc = np.append(mean_acc, np.mean(mean_acc))
    std_acc = np.append(std_acc, np.std(mean_acc, ddof=1))
    mean_itr = np.append(mean_itr, np.mean(mean_itr))
    std_itr = np.append(std_itr, np.std(mean_itr, ddof=1))

    # ==== LOSO格式csv输出 ====
    subjects = [f'Subject{i+1}' for i in range(final_acc_list.shape[1])]
    rows = []
    for j in range(final_acc_list.shape[1]):
        accs = final_acc_list[:, j]
        itrs = [calc_itr(acc, Nf, ws) for acc in accs]
        # LOSO下每个subject只会有一条acc
        rows.append({
            'Subject': subjects[j],
            'Acc (%)': f'{accs[0]*100:.2f}',
            'ITR (bits/min)': f'{itrs[0]:.2f}'
        })
    # 计算均值±方差
    all_accs = final_acc_list.flatten()
    all_itrs = [calc_itr(acc, Nf, ws) for acc in all_accs]
    rows.append({
        'Subject': 'Mean±Std',
        'Acc (%)': f'{np.mean(all_accs)*100:.2f}±{np.std(all_accs, ddof=1)*100:.2f}',
        'ITR (bits/min)': f'{np.mean(all_itrs):.2f}±{np.std(all_itrs, ddof=1):.2f}'
    })
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = f'../Result/{dataset}/{model_name}/{proportion}_{val_way}_Classification_Result({win_size}S).csv'
    ensure_dir_exists(csv_path)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


    data1 = final_mean_list

    # cal the length of data
    len_data = len(data1)

    # set x axis element for bar
    a = [i for i in range(len_data)]

    # set size of figure
    plt.figure(figsize=(20, 8), dpi=80)

    # adjust the direction of ticks
    matplotlib.rcParams['ytick.direction'] = 'in'

    # add grid for y axis
    # plt.rc('axes', axisbelow=True)  # 将网格始终置于底部
    # plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.4)

    # plot the bars
    for i in range(len_data):
        plt.bar(a[i], data1[i], width=0.35, label='final')

    # set ticks for x axis and y axis
    x_ticks_bound = [i for i in range(len_data)]
    x_ticks_content = [str(i + 1) for i in range(len_data - 1)]
    x_ticks_content.append('M')
    plt.xticks(x_ticks_bound, x_ticks_content, fontsize=10)

    y_ticks_bound = [i * 0.1 for i in range(11)]
    y_ticks_content = [str(i * 10) for i in range(11)]
    plt.yticks(y_ticks_bound, y_ticks_content, fontsize=15)

    # set label for data
    plt.xlabel('Subject', fontsize=15)
    plt.ylabel('Accuracy(%)', fontsize=15)
    plt.title(f'{model_name} {proportion} {val_way} Classification Result({win_size}S)', fontsize=15)

    # adjust size of axis
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    if text == True:
        # text elements on the top of bar
        for i in range(len_data):
           delta = 0.05
           if i != len_data - 1:
               plt.text(a[i] - delta * 3.5, data1[i] + delta * 0.1, f'{data1[i] * 100:.2f}')
           else:
               plt.text(a[i] - delta * 6.0, data1[i] + delta * 0.1,
                        f'{final_mean_list[-1] * 100:.2f}±{final_var_list[-1] * 100:.2f}', color='r')
    img_path = f'../Result/{dataset}/{model_name}/{proportion}_{val_way}_Classification_Result({win_size}S).png'
    ensure_dir_exists(img_path)
    plt.savefig(img_path)
    plt.show()
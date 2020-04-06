import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def box_plot(oscillation_features, impact_factor_column, feature_column, impact_factor_label, x_label, y_label):
    #for one features in one figure
    data_group = {}
    for l in impact_factor_label:
        data_group[l] = []
    for d in oscillation_features:
        data_group[d[impact_factor_column]].append(d[feature_column])
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    mean_value=[]
    for label in data_group.keys():
        data_points_y = data_group[label]
        mean_value.append((np.mean(data_points_y),#0 y-mean location
                 np.percentile(data_points_y,90),#1 y-90th location
                 np.percentile(data_points_y,10),#2 y-10th location
                 np.std(data_points_y),#3 y std
                 label,#4 label
                 data_points_y))#5 y samples
    for i in range(len(mean_value)):
        if i>=1:
            print(mean_value[i][4],round(mean_value[i][0], 2), mean_value[i-1][4], round(mean_value[i-1][0], 2), 'p-value', stats.ttest_ind(mean_value[i][5], mean_value[i-1][5])[1])
        
    plt.plot([i+1 for i in range(len(data_group.keys()))],[np.percentile(mv[5],50) for mv in mean_value],color='orange',linestyle='--',linewidth=1)
    box_data=[]
    for mv in mean_value:
        box_data.append(mv[5])
    plt.boxplot(box_data,whis=[5,95],labels=data_group.keys())
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.show()

    return mean_value

def bar_plot(oscillation_features, impact_factor_column, feature_column_group,
             impact_factor_group_label, feature_column_group_label, x_label, y_label, bar=True):
    #for multiple features in one figure
    data_group = {}
    for l in impact_factor_group_label:
        data_group[l] = {}
        for fl in feature_column_group_label:
            data_group[l][fl] = []
    for d in oscillation_features:
        for i in range(len(feature_column_group_label)):
            data_group[d[impact_factor_column]][feature_column_group_label[i]].append(d[feature_column_group[i]])

    color_group = ['r', 'g', 'b']
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    label_num = 0
    overall_mean_value = []
    for label in data_group.keys():
        mean_value = []
        data_points_y_group = data_group[label]
        for fl in data_points_y_group.keys():
            data_points_y = data_points_y_group[fl]
            mean_value.append((np.mean(data_points_y),  # 0 y-mean location
                               np.percentile(data_points_y, 90),  # 1 y-90th location
                               np.percentile(data_points_y, 10),  # 2 y-10th location
                               np.std(data_points_y),  # 3 y std
                               label,  # 4 label
                               data_points_y,  # 5 y samples
                               fl))  # 6 feature label

        for i in range(len(mean_value)):
            if i >= 1:
                print(mean_value[i][4], mean_value[i][6], round(mean_value[i][0], 2), mean_value[i - 1][6],
                      round(mean_value[i - 1][0], 2), 'p-value',
                      stats.ttest_ind(mean_value[i][5], mean_value[i - 1][5])[1])
                print('\n')
        plt.plot(np.arange(1, len(mean_value) + 1), [np.mean(mv[5]) for mv in mean_value],
                 linewidth=2, label=impact_factor_group_label[label_num], color=color_group[label_num], linestyle='-')
        plt.scatter(np.arange(1, len(mean_value) + 1), [np.mean(mv[5]) for mv in mean_value],
                    color=color_group[label_num])
        if bar:
            stick_width = .075
            stick_line_style = '--'
            stick_line_width = 1.5
            for i in range(len(mean_value)):
                line2_1 = plt.plot([i + 1, i + 1], [mean_value[i][1], mean_value[i][2]], color=color_group[label_num],
                                   linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][1], mean_value[i][1]],
                         color=color_group[label_num],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][2], mean_value[i][2]],
                         color=color_group[label_num],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
        label_num += 1
        overall_mean_value.append(mean_value)

    for j in range(len(overall_mean_value)):
        if j >= 1:
            for i in range(len(mean_value)):
                print(overall_mean_value[j][i][6], overall_mean_value[j][i][4], round(overall_mean_value[j][i][0], 2),
                      overall_mean_value[j-1][i][4], round(overall_mean_value[j-1][i][0], 2), 'p-value',
                      stats.ttest_ind(overall_mean_value[j][i][5], overall_mean_value[j-1][i][5])[1])
                print('\n')

    plt.legend(loc=2)
    plt.xticks(np.arange(1, len(mean_value) + 1), feature_column_group_label, fontsize=12)
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.show()


def bar_plot_inverse(oscillation_features, impact_factor_column, feature_column_group,
             impact_factor_group_label, feature_column_group_label, x_label, y_label, bar=True):
    #for multiple features in one figure
    data_group = {}
    for fl in feature_column_group_label:
        data_group[fl] = {}
        for l in impact_factor_group_label:
            data_group[fl][l] = []

    for d in oscillation_features:
        for i in range(len(feature_column_group_label)):
            data_group[feature_column_group_label[i]][d[impact_factor_column]].append(d[feature_column_group[i]])

    color_group = ['r', 'g', 'b']
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    label_num = 0
    overall_mean_value = []
    for label in data_group.keys():
        mean_value = []
        data_points_y_group = data_group[label]
        for fl in data_points_y_group.keys():
            data_points_y = data_points_y_group[fl]
            mean_value.append((np.mean(data_points_y),  # 0 y-mean location
                               np.percentile(data_points_y, 90),  # 1 y-90th location
                               np.percentile(data_points_y, 10),  # 2 y-10th location
                               np.std(data_points_y),  # 3 y std
                               label,  # 4 label
                               data_points_y,  # 5 y samples
                               fl))  # 6 feature label

        for i in range(len(mean_value)):
            if i >= 1:
                print(mean_value[i][4], mean_value[i][6], round(mean_value[i][0], 2), mean_value[i - 1][6],
                      round(mean_value[i - 1][0], 2), 'p-value',
                      stats.ttest_ind(mean_value[i][5], mean_value[i - 1][5])[1])
                print('\n')
        plt.plot(np.arange(1, len(mean_value) + 1), [np.mean(mv[5]) for mv in mean_value],
                 linewidth=2, label=feature_column_group_label[label_num], color=color_group[label_num], linestyle='-')
        plt.scatter(np.arange(1, len(mean_value) + 1), [np.mean(mv[5]) for mv in mean_value],
                    color=color_group[label_num])
        if bar:
            stick_width = .075
            stick_line_style = '--'
            stick_line_width = 1.5
            for i in range(len(mean_value)):
                line2_1 = plt.plot([i + 1, i + 1], [mean_value[i][1], mean_value[i][2]], color=color_group[label_num],
                                   linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][1], mean_value[i][1]],
                         color=color_group[label_num],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][2], mean_value[i][2]],
                         color=color_group[label_num],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
        label_num += 1
        overall_mean_value.append(mean_value)

    for j in range(len(overall_mean_value)):
        if j >= 1:
            for i in range(len(mean_value)):
                print(overall_mean_value[j][i][6], overall_mean_value[j][i][4], round(overall_mean_value[j][i][0], 2),
                      overall_mean_value[j-1][i][4], round(overall_mean_value[j-1][i][0], 2), 'p-value',
                      stats.ttest_ind(overall_mean_value[j][i][5], overall_mean_value[j-1][i][5])[1])
                print('\n')

    plt.legend(loc=1)
    plt.xticks(np.arange(1, len(mean_value) + 1), impact_factor_group_label)
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.show()
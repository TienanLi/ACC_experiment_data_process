import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base_functions import linear_regression, time_of_week_to_hms, exclude_outlier
from matplotlib import rc
from scipy import stats

font = {'family': 'DejaVu Sans',
        'size': 16}
rc('font', **font)

def q_k_figure(density1,volumn1,density2,volumn2,label1,label2,folder_name,setting):
    print('\n')
    free_flow_speed = 70  # mph
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.7])
    plt.scatter(density1, volumn1, s=8, c='r', marker='o', label=label1)
    plt.scatter(density2, volumn2, s=8, c='b', marker='o', label=label2)
    plt.legend()
    plt.title('q-k: headway setting' + setting)
    plt.xlabel('density (veh/mile)')
    plt.ylabel('volumn (veh/h)')
    plt.xlim([0, 150])
    plt.ylim([0, max(volumn1) * 1.5])
    coef, intercept, p_value = linear_regression(density1, volumn1)
    print(label1, 'q-k coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    critical_d = intercept / (free_flow_speed - coef)
    capacity = free_flow_speed * critical_d
    print('capacity:', round(capacity, 0))
    plt.plot([critical_d, 150], [critical_d * coef + intercept, 150 * coef + intercept], 'r--')
    plt.plot([0, critical_d], [0, critical_d * free_flow_speed], 'k--')
    coef, intercept, p_value = linear_regression(density2, volumn2)
    print(label2, 'q-k coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    critical_d = intercept / (free_flow_speed - coef)
    capacity = free_flow_speed * critical_d
    print('capacity:', round(capacity, 0))
    plt.plot([critical_d, 150], [critical_d * coef + intercept, 150 * coef + intercept], 'b--')
    plt.plot([0, critical_d], [0, critical_d * free_flow_speed], 'k--')
    plt.savefig(os.path.dirname(__file__) + folder_name + 'q-k headway setting' + setting +'.png')

def s_v_figure(speed1,spacing1,speed2,spacing2,label1,label2,folder_name,setting):
    print('\n')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.7])
    plt.scatter(speed1 * 0.44704, spacing1, s=24, c='r', marker='o', label=label1)
    plt.scatter(speed2 * 0.44704, spacing2, s=24, c='b', marker='o', label=label2)
    plt.legend()
    plt.title('s-v: headway setting' + setting)
    plt.xlabel('speed (m/s)')
    plt.ylabel('spacing (m)')
    plt.xlim([0, 35])
    plt.ylim([0, max(spacing2) * 1.5])
    coef, intercept, p_value = linear_regression(speed1 * 0.44704, spacing1)
    print(label1, 's-v coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    plt.plot([0, 35], [0 * coef + intercept, 35 * coef + intercept], 'r--')
    coef, intercept, p_value = linear_regression(speed2 * 0.44704, spacing2)
    print(label2, 's-v coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    plt.plot([0, 35], [0 * coef + intercept, 35 * coef + intercept], 'b--')
    plt.savefig(os.path.dirname(__file__) + folder_name + 's-v headway setting' + setting +'.png')

def hour_min_to_min(time_point):
    return time_point[0] * 60 + time_point[1]

def read_data_from_summary_csv(folder_name, headway):
    traj_dict=[]
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if 'headway_'+headway not in csv_file:
            continue
        traj_df = pd.read_csv(os.path.dirname(__file__)+folder_name+'/' + csv_file, header = None)
        traj_df[10] = traj_df[4] - traj_df[5] # ACC1 spacing
        traj_df[11] = traj_df[5] - traj_df[6] # ACC2 spacing
        traj_dict.append(traj_df)
    return traj_dict

def read_data_from_summary_csv_overall(folder_name, headway_period, date):
    traj_dict=[]
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if 'summary' not in csv_file:
            continue
        traj_df = pd.read_csv(os.path.dirname(__file__)+folder_name+'/' + csv_file, header = None)
        traj_df[10] = traj_df[4] - traj_df[5] # ACC1 spacing
        traj_df[11] = traj_df[5] - traj_df[6] # ACC2 spacing
        if date > 20:
            EST = -5
        else:
            EST = -4
        traj_df[12], traj_df[13], traj_df[14] = time_of_week_to_hms(traj_df[0], EST)
        traj_df[15] = hour_min_to_min([traj_df[12], traj_df[13]])
        for period in headway_period:
            start = period[0]
            end = period[1]
            traj_df_split = traj_df[(traj_df[15]>=hour_min_to_min(start))
                                    & (traj_df[15]<=hour_min_to_min(end))]
            if traj_df_split.size > 0:
                traj_dict.append(traj_df_split)
    return traj_dict

def find_equilibrium(traj_dict, date):
    TH = get_TH(date)
    equilibrium_status_ACC1 = []
    equilibrium_status_ACC2 = []
    ACC1_column = [1, 10]
    ACC2_column = [2, 11]
    TH_horizon = 100 #horizon - in frequency
    for traj_df in traj_dict:
        traj_df = traj_df.reset_index()
        #ACC1
        equilibrium_end_point = traj_df.index[0]
        for i in range(traj_df.index[0], traj_df.index[-1]-TH_horizon):
            if i < equilibrium_end_point:
                continue
            variation = [max(abs(traj_df[c][i:i+TH_horizon] - traj_df[c][i])) for c in ACC1_column]
            equlibrium = True
            for c in range(len(TH)):
                if variation[c] > TH[c]:
                    equlibrium = False
            if equlibrium:
                x = TH_horizon
                while equlibrium:
                    for j in range(len(TH)):
                        if abs(traj_df[ACC1_column[j]][i+x] - traj_df[ACC1_column[j]][i]) > TH[j]:
                            equlibrium = False
                    x += 1
                    if i + x > traj_df.index[-1]:
                        break
                equilibrium_status_ACC1.append([np.mean(traj_df[c][i:i+x])
                                                  for c in ACC1_column])
                traj_df[:][i:i + x].to_csv('ACC1_equilibrium%s_%s.csv'%(date,traj_df[0][i]))
                equilibrium_end_point = i + x
        #ACC2
        equilibrium_end_point = traj_df.index[0]
        for i in range(traj_df.index[0], traj_df.index[-1]-TH_horizon):
            if i < equilibrium_end_point:
                continue
            variation = [max(abs(traj_df[c][i:i+TH_horizon] - traj_df[c][i])) for c in ACC2_column]
            equlibrium = True
            for c in range(len(TH)):
                if variation[c] > TH[c]:
                    equlibrium = False
            if equlibrium:
                x = TH_horizon
                while equlibrium:
                    for j in range(len(TH)):
                        if abs(traj_df[ACC2_column[j]][i+x] - traj_df[ACC2_column[j]][i]) > TH[j]:
                            equlibrium = False
                    x += 1
                    if i + x > traj_df.index[-1]:
                        break
                equilibrium_status_ACC2.append([np.mean(traj_df[c][i:i+x])
                                                  for c in ACC2_column])
                traj_df[:][i:i + x].to_csv('ACC2_equilibrium%s_%s.csv'%(date,traj_df[0][i]))
                equilibrium_end_point = i + x
    ACC1 = pd.DataFrame(data=equilibrium_status_ACC1)
    ACC2 = pd.DataFrame(data=equilibrium_status_ACC2)
    ACC1 = exclude_outlier(ACC1[(ACC1[0] > 5) & (ACC1[0] < 70)].sort_values(by=[0]))
    ACC2 = exclude_outlier(ACC2[(ACC2[0] > 5) & (ACC2[0] < 70)].sort_values(by=[0]))
    print('ACC1', len(ACC1))
    print('ACC2', len(ACC2))
    return ACC1, ACC2

def read_data_from_equlirbium_csv(folder_name, headway_period):
    equilibrium_status_ACC1 = []
    equilibrium_status_ACC2 = []
    ACC1_column = ['1', '10']
    ACC2_column = ['2', '11']
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if 'equilibrium' not in csv_file:
            continue
        traj_df = pd.read_csv(os.path.dirname(__file__)+folder_name+'/' + csv_file)
        for period in headway_period:
            start = period[0]
            end = period[1]
            traj_df_split = traj_df[(traj_df['15']>=hour_min_to_min(start))
                                    & (traj_df['15']<=hour_min_to_min(end))]
            if len(traj_df_split) == 0:
                continue
            if 'ACC1' in csv_file:
                equilibrium_status_ACC1.append([np.nanmean(traj_df_split[c]) for c in ACC1_column])
            if 'ACC2' in csv_file:
                equilibrium_status_ACC2.append([np.nanmean(traj_df_split[c]) for c in ACC2_column])
    ACC1 = pd.DataFrame(data=equilibrium_status_ACC1)
    ACC2 = pd.DataFrame(data=equilibrium_status_ACC2)
    ACC1 = exclude_outlier(ACC1[(ACC1[0] > 5) & (ACC1[0] < 70)].sort_values(by=[0]))
    ACC2 = exclude_outlier(ACC2[(ACC2[0] > 5) & (ACC2[0] < 70)].sort_values(by=[0]))
    print('ACC1 all:', len(ACC1))
    print('ACC2 all:', len(ACC2))
    return ACC1, ACC2

def get_headway_period():
    headway_1_period = {}
    headway_3_period = {}
    #Tesla 3/12
    headway_1_period[12] = [((9,55),(10,25)), ((11,30),(11,50)), ((12,15),(12,25)), ((14,5),(14,35))]
    headway_3_period[12] = [((10,30),(10,55)), ((11,50),(12,15)) , ((13,50),(14,3)), ((22,0),(22,17))]
    #Civic Normal 3/14
    headway_1_period[14] = [((9,40),(10,8)), ((11,45),(12,4)), ((13,58),(14,40))]
    headway_3_period[14] = [((10,9),(10,40)), ((12,5),(12,50)) , ((13,40),(13,57))]
    # Civic Sport 3/15
    headway_1_period[15] = [((9, 48), (10, 14)), ((11, 35), (12, 8)), ((14, 41), (15, 20))]
    headway_3_period[15] = [((10, 15), (10, 50)), ((12, 9), (12, 40)), ((13, 45), (14, 40))]
    #Tesla 2/21
    headway_1_period[21] = [((13, 9), (13, 31)),((14, 16), (14, 40))]
    headway_3_period[21] = []
    #Tesla 2/22
    headway_1_period[22] = [((12, 0), (13, 18))]
    headway_3_period[22] = [((13, 20), (14, 50))]
    #Tesla 2/23
    headway_1_period[23] = [((11, 30), (12, 35))]
    headway_3_period[23] = [((11, 15), (11, 27))]

    return headway_1_period, headway_3_period

def get_TH(date):
    if date in [14, 15]:
        return [1.5, 1.5]
    else:
        return [1, 1]

def get_FD_parameters(ACC1, ACC2):
    spacing1 = np.array(ACC1[1])  # m
    speed1 = np.array(ACC1[0])  # in mph
    density1 = 1609.344 / spacing1  # veh/mile
    volumn1 = speed1 * density1  # veh/h
    spacing2 = np.array(ACC2[1])
    speed2 = np.array(ACC2[0])  # in mph
    density2 = 1609.344 / spacing2
    volumn2 = speed2 * density2
    return spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2

def draw_FD(ACC1, ACC2, setting, folder_name, label1 = 'ACC1', label2 = 'ACC2'):
    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC1, ACC2)
    q_k_figure(density1, volumn1, density2, volumn2, label1, label2, folder_name, setting)
    s_v_figure(speed1,spacing1,speed2,spacing2,label1,label2,folder_name,setting)

def main_single_day(folder_name, date):
    ACC_headway = {}
    headway_setting = ['3', '1']
    headway_1_period, headway_3_period = get_headway_period()
    headway_period = {'1': headway_1_period[date], '3': headway_3_period[date]}
    for headway in headway_setting:
        print('\nheadway',headway)
        if len(headway_period[headway]) == 0:
            continue
        # traj_dict = read_data_from_summary_csv_overall(folder_name, headway_period[headway], date)
        # traj_dict = read_data_from_summary_csv(folder_name+'split_traj/', headway)
        # equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date)
        equilibrium_status_ACC1, equilibrium_status_ACC2 = \
            read_data_from_equlirbium_csv('/platooned_data/03-%s-2020/equilibrium_traj/' % (str(date).zfill(2)),
                                          headway_period[headway])

        draw_FD(equilibrium_status_ACC1, equilibrium_status_ACC2, headway, folder_name)
        ACC_headway[headway] = pd.concat([equilibrium_status_ACC1, equilibrium_status_ACC2])
    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])
    s_v_figure(speed1,spacing1,speed2,spacing2,'headway 1','headway 3',folder_name,'-combined')
    q_k_figure(density1, volumn1, density2, volumn2,'headway 1','headway 3',folder_name,'-combined')

def main_multi_day(date_group):
    headway_setting = ['3', '1']
    headway_1_period, headway_3_period = get_headway_period()
    ACC1 = pd.DataFrame()
    ACC2 = pd.DataFrame()
    for headway in headway_setting:
        print('\nheadway',headway)
        for date in date_group:
            print(date)
            headway_period = {'1': headway_1_period[date], '3': headway_3_period[date]}
            if len(headway_period[headway]) == 0:
                continue
            # traj_dict = read_data_from_summary_csv_overall('/platooned_data/02-%s-2020/'%(str(date).zfill(2)),
            #                                                headway_period[headway], date)
            # equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date)
            equilibrium_status_ACC1, equilibrium_status_ACC2 = \
                read_data_from_equlirbium_csv('/platooned_data/02-%s-2020/equilibrium_traj/'%(str(date).zfill(2)),
                                                           headway_period[headway])
            ACC1 = pd.concat([ACC1, equilibrium_status_ACC1])
            ACC2 = pd.concat([ACC2, equilibrium_status_ACC2])
        ACC1 = exclude_outlier(ACC1.sort_values(by=[0]))
        ACC2 = exclude_outlier(ACC2.sort_values(by=[0]))
        print('ACC1 all:',len(ACC1))
        print('ACC2 all:',len(ACC2))
        draw_FD(ACC1, ACC2, headway, '/platooned_data/02-23-2020/')


if __name__=='__main__':
    Multi = True
    experiment_date = [21,22,23]

    if Multi:
        main_multi_day(experiment_date)
    else:
        for date in experiment_date:
            print(date)
            traj_dict = main_single_day('/platooned_data/02-%s-2020/'%(str(date).zfill(2)), date)

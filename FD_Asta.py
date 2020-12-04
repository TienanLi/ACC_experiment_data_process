import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from base_functions import linear_regression

def vehicles_column():
    # self_speed, spacing, lead_speed， elevation
    return [[8, 36, 1, 7], [15, 37, 8, 14], [22, 38, 15, 21], [29, 39, 22, 28]]

def veh_length():
    #2018 audi A8, 2019 Tesla Model 3, 2018 BMW X5, 2019 Mercedes A Class, 2018 Audi A6
    return [5.27, 4.69, 4.89, 4.55, 4.93] #in m

def find_equilibrium(traj_dict):
    TH = [0.44704, 1, 0.44704] #speed spacing
    TH_horizon = 100 #horizon - in frequency

    speed_spacing_column_set = vehicles_column()
    for k, traj_df in traj_dict.items():
        traj_df = traj_df.reset_index()
        for veh in range(4):
            speed_spacing_column = speed_spacing_column_set[veh]
            equilibrium_end_point = traj_df.index[0]

            for i in range(traj_df.index[0], traj_df.index[-1] - TH_horizon):
                if i < equilibrium_end_point:
                    continue

                # naive threshold for to keep away from disturbances
                if i > TH_horizon:
                    if max([max(traj_df[c][i - TH_horizon:i]) - min(traj_df[c][i - TH_horizon:i]) \
                            for c in [speed_spacing_column[0],speed_spacing_column[2]]]) > 3 * 0.44704:
                        continue

                variation = [max(traj_df[c][i:i + TH_horizon]) - min(traj_df[c][i:i + TH_horizon]) for c in
                             speed_spacing_column]
                equlibrium = True
                for c in [1, 0, 2]:
                    if variation[c] > TH[c]:
                        equlibrium = False
                speed_diff = abs(traj_df[speed_spacing_column[0]][i:i + TH_horizon] - \
                                 traj_df[speed_spacing_column[2]][i:i + TH_horizon])
                if (max(speed_diff) > TH[0]) and equlibrium:  # speed_diff_threshold
                    equlibrium = False
                if equlibrium:
                    x = TH_horizon
                    maxDict = [max(traj_df[c][i:i + x]) for c in speed_spacing_column]
                    minDict = [min(traj_df[c][i:i + x]) for c in speed_spacing_column]
                    while equlibrium:

                        for j in [1, 0, 2]:
                            maxDict[j] = max(traj_df[speed_spacing_column[j]][i + x], maxDict[j])
                            minDict[j] = min(traj_df[speed_spacing_column[j]][i + x], minDict[j])

                            # if abs(traj_df[speed_spacing_column[j]][i + x] - \
                            #        traj_df[speed_spacing_column[j]][i]) > TH[j]:
                            if maxDict[j] - minDict[j] > TH[j]:
                                equlibrium = False
                        if abs(traj_df[speed_spacing_column[0]][i + x] - \
                               traj_df[speed_spacing_column[2]][
                            i + x]) > TH[0]:
                            equlibrium = False
                        x += 1
                        if i + x > traj_df.index[-1]:
                            break
                    equilibrium_end_point = i + x

                    # exclude the singularity condition
                    # if len(find_peaks(traj_df[speed_spacing_column[1]][i:i + x],
                    #                   prominence=.05, width=(0, 100))[0]) + \
                    #         len(find_peaks(-traj_df[speed_spacing_column[1]][i:i + x],
                    #                        prominence=.05, width=(0, 100))[0]) == 1:
                    #     continue

                    traj_df[[0] + speed_spacing_column][i:i + x].to_csv(os.getcwd() + \
                                        '/platooned_data/Asta_data/equilibrium_traj/equilibrium%s_%s_%s_%s_p%s.csv'
                                           %(veh+2, traj_df[speed_spacing_column[0]][i],
                                             traj_df[0][i], traj_df[0][i + x - 1], k))

def read_data_from_equlirbium_csv(folder_name, veh, min_speed=5, max_speed=30):
    speed_spacing_column_set = vehicles_column()
    speed_spacing_column = [str(c) for c in speed_spacing_column_set[veh - 2]]
    experiment = {'min':[]}
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if 'equilibrium' + str(veh) not in csv_file:
            continue
        experimentSet = csv_file.split('p')[1].split('.')[0]
        setSetting = group_setting(experimentSet)
        if setSetting is None:
            continue
        if setSetting == 'max':
            continue
        traj_df = pd.read_csv(os.path.dirname(__file__)+folder_name+'/' + csv_file)

        elevation = np.array(traj_df[speed_spacing_column[-1]])
        speed = np.array(traj_df[speed_spacing_column[0]])
        slope = np.mean((elevation[1:] - elevation[:-1]) / (speed[:-1] * 0.1)) #in the unit of vertical/horizontal

        experiment[setSetting].append([np.mean(traj_df[c])
                                        for c in speed_spacing_column[:-1]] +
                                       [traj_df['0'][0],
                                        traj_df['0'][len(traj_df)-1],
                                        experimentSet, slope])
    equil_DF = {k: pd.DataFrame(data=v).sort_values(by=[0]) for k, v in experiment.items()}
    equil_DF ={k: v[(v[0] > min_speed) & (v[0] < max_speed)] for k, v in equil_DF.items()}
    return equil_DF

def read_data_from_csv(folder_name):
    relevant_col = list(set([item for L in vehicles_column() for item in L])) + [0]
    traj_dict = {}
    for csv_file in os.listdir(os.getcwd() + folder_name):
        if 'platoon' not in csv_file:
            continue
        traj_df = pd.read_csv(os.getcwd() + folder_name + csv_file, header=None, engine='python')
        traj_dict[csv_file.split('platoon')[1].split('.')[0]] = traj_df[6:][relevant_col].astype('float32')
    return traj_dict

def group_setting(num):
    setting = {'1': 'min', '2': 'min', '3': None, '4': 'max', '5': 'min',
               '6': 'min', '7': 'min', '8': 'min', '9': 'min', '10': None}
    return setting[num]

def get_FD_parameters(equil_DF):
    spacing = np.array(equil_DF[1])  # m
    speed = np.array(equil_DF[0])  # in m/s
    density = 1609.344 / spacing  # veh/mile
    volume = speed / 0.44704 * density  # veh/h
    return spacing, speed, density, volume

def q_k_figure():
    pass

def s_v_figure(speed, spacing, setting, label):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.7])
    plt.scatter(speed, spacing, s=24, c='r', marker='o')
    plt.title(label + ' s-v: headway setting ' + setting)
    plt.xlabel('speed (m/s)')
    plt.ylabel('spacing (m)')
    plt.xlim([0, 35])
    plt.ylim([0, max(spacing) * 1.5])
    coef, intercept, p_value = linear_regression(speed, spacing, weight=None)
    print(label, 's-v coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    plt.plot([0, 35], [0 * coef + intercept, 35 * coef + intercept], 'r--')
    plt.savefig(os.getcwd() + '/platooned_data/Asta_data/veh %s s-v headway setting '%label + setting + '.png')

def draw_FD(equil_DF, setting, veh):
    print('\n', veh, setting)
    if len(equil_DF) <= 1:
        print('less than one sample for linear regression')
        return
    spacing, speed, density, volume = get_FD_parameters(equil_DF)
    s_v_figure(speed, spacing, setting, label=str(veh))
    # q_k_figure(density, volumn, density, volumn, setting)

def equilibrium_analysis():
    for veh in range(2, 6):
        equil_DF = read_data_from_equlirbium_csv('/platooned_data/Asta_data/equilibrium_traj/', veh)
        for s in equil_DF:
            equil_DF[s][1] += veh_length()[veh - 2] # from inter-vehicle spacing to bumper-bumper spacing
        fo = open(os.path.dirname(__file__) + '/platooned_data/Asta_data/equilibrium_traj/FD_data_Asta_veh%s'%veh, 'wb')
        pickle.dump(equil_DF, fo)
        fo.close()
        for k, v in equil_DF.items():
            if k == 'max':
                continue
            draw_FD(v, k, veh)

def main():
    # traj_dict = read_data_from_csv('/platooned_data/Asta_data/')
    # find_equilibrium(traj_dict)

    equilibrium_analysis()


if __name__=='__main__':
    main()
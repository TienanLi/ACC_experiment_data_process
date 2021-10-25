import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from base_functions import linear_regression, denoise_grade, moving_average

def vehicles_column(veh):
    # self_speed, spacing, lead_speed
    return ['Speed%s'%(str(int(veh + 1))), 'IVS%s(m)'%(str(int(veh))), 'Speed%s'%(str(int(veh))), ]

def veh_length():
    #2018 audi A8, 2019 Tesla Model 3, 2018 BMW X5, 2019 Mercedes A Class, 2018 Audi A6
    return {'Audi(A8)':5.27, 'Tesla(Model3)':4.69, 'BMW(X5)':4.89, 'Mercedes(AClass)':4.55, 'Audi(A6)':4.93} #in m

def get_veh_name(veh):
    vehNameDict = {1:'Audi(A8)', 2:'Tesla(Model3)', 3:'BMW(X5)', 4:'Mercedes(AClass)', 5:'Audi(A6)'}
    return vehNameDict[veh]

def find_equilibrium(traj_dict, vehicleInfoDict, statusDict):
    TH = [0.44704, 1, 0.44704] #self_speed (m/s), spacing(m), lead_speed(m/s)
    TH_horizon = 100 #horizon - in frequency
    disturbance_away = 100

    # TH = [0.25, 0.25, 0.25] #self_speed (m/s), spacing(m), lead_speed(m/s)
    # TH_horizon = 100 #horizon - in frequency
    # disturbance_away = 100

    for k, traj_df in traj_dict.items():
        traj_df = traj_df.reset_index()
        vehicleInfo = vehicleInfoDict[k]
        if vehicleInfo[3] == '0': #human driving
            continue
        for veh in range(1, 5):
            vehName = vehicleInfo[1][veh+1]
            vehLeader = vehicleInfo[1][veh]
            speed_spacing_column = vehicles_column(veh)
            equilibrium_end_point = traj_df.index[0]

            for i in range(traj_df.index[0], traj_df.index[-1] - TH_horizon):
                if i < equilibrium_end_point:
                    continue
                if statusDict[k]['Driver%s'%(str(int(veh + 1)))][i] != 'ACC':
                    continue

                # naive threshold for to keep away from disturbances
                if i > TH_horizon:
                    if max([max(traj_df[c][i - disturbance_away:i]) - min(traj_df[c][i - disturbance_away:i]) \
                            for c in [speed_spacing_column[0],speed_spacing_column[2]]]) > 3 * 0.44704:
                        continue

                # lead_speed, self_speed, and spacing threshold
                variation = [max(traj_df[c][i:i + TH_horizon]) - min(traj_df[c][i:i + TH_horizon]) for c in
                             speed_spacing_column]
                equlibrium = True
                for c in [1, 0, 2]:
                    if variation[c] > TH[c]:
                        equlibrium = False

                # speed_diff_threshold
                speed_diff = abs(traj_df[speed_spacing_column[0]][i:i + TH_horizon] - \
                                 traj_df[speed_spacing_column[2]][i:i + TH_horizon])
                if (max(speed_diff) > TH[0]) and equlibrium:
                    equlibrium = False

                if equlibrium:
                    x = TH_horizon
                    maxDict = [max(traj_df[c][i:i + x]) for c in speed_spacing_column]
                    minDict = [min(traj_df[c][i:i + x]) for c in speed_spacing_column]
                    while equlibrium:

                        for j in [1, 0, 2]:
                            # lead_speed, self_speed, and spacing
                            maxDict[j] = max(traj_df[speed_spacing_column[j]][i + x], maxDict[j])
                            minDict[j] = min(traj_df[speed_spacing_column[j]][i + x], minDict[j])
                            if maxDict[j] - minDict[j] > TH[j]:
                                equlibrium = False

                        #speed diff threshold
                        if abs(traj_df[speed_spacing_column[0]][i + x] - \
                               traj_df[speed_spacing_column[2]][i + x]) > TH[0]:
                            equlibrium = False
                        x += 1
                        if i + x > traj_df.index[-1]:
                            break

                        if statusDict[k]['Driver%s' % (str(int(veh + 1)))][i] != 'ACC':
                            break

                    equilibrium_end_point = i + x

                    traj_df[['Time'] + speed_spacing_column][i:i + x].to_csv(os.getcwd() + \
                                        '/platooned_data/Asta_data/equilibrium_traj/equilibrium_%s_%s_%s_%s_%s_p%s.csv'
                                           %(vehName, vehLeader, round(traj_df[speed_spacing_column[0]][i], 1),
                                             traj_df['Time'][i], traj_df['Time'][i + x - 1], k))

def calulate_road_grade(experiment, vehiclePosition, timeStart, timeEnd):
    #period to consider for grade constraint
    before = 10 #s
    after = 0 #s

    for csv_file in os.listdir(os.getcwd() + '/platooned_data/ASta_data/'):
        if str(experiment) + '.csv' not in csv_file:
            continue
        infoRow = 5
        traj_df = pd.read_csv(os.getcwd() + '/platooned_data/ASta_data/' + csv_file,
                              header=infoRow, engine='python', sep=',')
        relevantCol = ['Time'] + ['Alt' + str(vehiclePosition)] + ['Speed' + str(vehiclePosition)]
        trajDict = traj_df[:][relevantCol].astype('float32')
        trajDict = trajDict[(trajDict['Time'] > timeStart - before) & (trajDict['Time'] < timeEnd + after)]

    grade = trajDict['Alt' + str(vehiclePosition)].diff() / trajDict['Speed' + str(vehiclePosition)] / .1 * 100  # %
    maxGrade = np.nanmax(abs(grade))
    return maxGrade


def read_data_from_equlirbium_csv(folder_name, veh, min_speed=5, max_speed=30):
    # speed_spacing_column_set = vehicles_column()
    # speed_spacing_column = [str(c) for c in speed_spacing_column_set[veh - 2]]
    experiment = {'min':[]}
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if 'equilibrium_' + get_veh_name(veh) not in csv_file:
            continue
        experimentSet = csv_file.split('p')[1].split('.')[0]
        setSetting = group_setting(experimentSet)
        if setSetting is None:
            continue
        if setSetting == 'max':
            continue
        traj_df = pd.read_csv(os.path.dirname(__file__)+folder_name+'/' + csv_file)

        # already filtered out
        # maxGrade = calulate_road_grade(experimentSet, veh, float(csv_file.split('_')[4]), float(csv_file.split('_')[5]))
        # if maxGrade > 3.5: #no more than 3% grade for the found equilibrium
        #     print(csv_file, maxGrade)
        #     continue

        leader = csv_file.split('_')[2]
        experiment[setSetting].append([np.mean(traj_df.iloc[:, 2]), # self_speed
                                       np.mean(traj_df.iloc[:, 3]) + veh_length()[leader], # spacing with leader length
                                       np.mean(traj_df.iloc[:, 4])] # lead_speed
                                       + [traj_df['Time'][0],
                                        traj_df['Time'][len(traj_df)-1],
                                        experimentSet])
    equil_DF = {k: pd.DataFrame(data=v).sort_values(by=[0]) for k, v in experiment.items()}
    equil_DF ={k: v[(v[0] > min_speed) & (v[0] < max_speed)] for k, v in equil_DF.items()}
    return equil_DF

def get_name(filename):
        return filename.split('platoon')[1].split('.')[0]

def read_data_from_csv(folder_name):
    trajDict = {}
    vehicleInfoDict = {}
    statusDict = {}
    for csv_file in os.listdir(os.getcwd() + folder_name):
        if 'platoon' not in csv_file:
            continue
        infoRow = 5

        platoonName = get_name(csv_file)
        info = pd.read_csv(os.getcwd() + folder_name + csv_file, header=None, sep=',',engine='python',
                              names=np.arange(12), nrows=infoRow, index_col=False)
        numOfVehicle = int(info.iloc[2][1])
        vehicleOrder = info.iloc[1][1 : numOfVehicle + 1]
        accMode = info.iloc[3][1]
        if infoRow == 5:
            headway = info.iloc[4][1]
        else:
            headway = 'min'
        vehicleInfoDict[platoonName] = [numOfVehicle, vehicleOrder, accMode, headway]

        traj_df = pd.read_csv(os.getcwd() + folder_name + csv_file, header=infoRow, engine='python', sep=',')
        relevantCol = ['Time'] + [col for col in traj_df.columns if 'Speed' in col] + [col for col in traj_df.columns if 'IVS' in col]
        trajDict[platoonName] = traj_df[:][relevantCol].astype('float32')
        statusCol = ['Time'] + [col for col in traj_df.columns if 'Driver' in col]
        statusDict[platoonName] = traj_df[:][statusCol]
        statusDict[platoonName]['Time'] = statusDict[platoonName]['Time'].astype('float32')

    return trajDict, vehicleInfoDict, statusDict

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
    plt.scatter(speed, spacing, s=24, c='C0', marker='o')
    plt.xlabel('speed (m/s)')
    plt.ylabel('spacing (m)')
    plt.xlim([0, 35])
    plt.ylim([0, max(spacing) * 1.5])
    coef, intercept, p_value, R2 = linear_regression(speed, spacing, weight=None, complete_return=True)
    regressionResult = 'coef: %s, intercept: %s, size: %s, $R^2$: %s'%(coef, intercept, len(speed), R2) if p_value < 0.1 \
        else 'Results not significant, size: %s, $R^2$: %s'%(len(speed), R2)
    print(label, 's-v coef:', coef, 'intercept:', intercept, 'p-value:', round(p_value, 3))
    plt.title(label + ' s-v: headway setting ' + setting + '\n' + regressionResult)
    plt.plot([0, 35], [0 * coef + intercept, 35 * coef + intercept], 'C0--')
    plt.savefig(os.getcwd() + '/platooned_data/Asta_data/veh %s s-v headway setting '%label + setting + '.png')

def draw_FD(equil_DF, setting, veh):
    print('\n', veh, setting)
    if len(equil_DF) <= 1:
        print('less than one sample for linear regression')
        return
    spacing, speed, density, volume = get_FD_parameters(equil_DF)

    s_v_figure(speed, spacing, setting, label= 'Asta ' + get_veh_name(veh))
    # q_k_figure(density, volumn, density, volumn, setting)

def equilibrium_analysis():
    for veh in range(2, 6):
        equil_DF = read_data_from_equlirbium_csv('/platooned_data/Asta_data/equilibrium_traj/', veh)
        fo = open(os.path.dirname(__file__) + \
                  '/platooned_data/Asta_data/equilibrium_traj/FD_data_Asta_veh_%s'%get_veh_name(veh), 'wb')
        pickle.dump(equil_DF, fo)
        fo.close()
        for k, v in equil_DF.items():
            if k == 'max':
                continue
            draw_FD(v, k, veh)

def main():
    # trajDict, vehicleInfoDict, statusDict = read_data_from_csv('/platooned_data/Asta_data/')
    # find_equilibrium(trajDict, vehicleInfoDict, statusDict)
    equilibrium_analysis()


if __name__=='__main__':
    main()
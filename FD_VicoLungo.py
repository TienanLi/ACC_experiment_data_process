import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from base_functions import linear_regression, denoise_grade, moving_average

def vehicles_column(veh):
    # self_speed, spacing, lead_speed
    return ['Speed%s'%(str(int(veh + 1))), 'IVS%s'%(str(int(veh))), 'Speed%s'%(str(int(veh))), ]

def veh_length():
    return {'Mitsubishi(SpaceStar)':3.80, 'KIA(Niro)':4.36, 'Ford(S-Max)':4.80, 'VW(GolfE)':4.25,
            'Mitsubishi(OutlanderPHEV)':4.69, 'Peugeot(3008GTLine)':4.45, 'Mini(Cooper)':3.93} #in m

def get_veh_name(veh):
    vehNameDict = {1: 'Mitsubishi(SpaceStar)', 2: 'KIA(Niro)', 3: 'Ford(S-Max)', 4: 'VW(GolfE)',
            5: 'Mitsubishi(OutlanderPHEV)', 6: 'Peugeot(3008GTLine)', 7: 'Mini(Cooper)'}
    return vehNameDict[veh]

def find_equilibrium(traj_dict, vehicleInfoDict, statusDict):
    TH = [0.44704, 1, 0.44704] #self_speed (m/s), spacing(m), lead_speed(m/s)
    TH_horizon = 100 #horizon - in frequency
    disturbance_away = 100


    for k, traj_df in traj_dict.items():
        traj_df = traj_df.reset_index()
        vehicleInfo = vehicleInfoDict[k]
        if vehicleInfo[3] == '0': #human driving
            continue
        for veh in range(1, vehicleInfo[0]):
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
                                        '/platooned_data/VicoLungo_data/equilibrium_traj/equilibrium_%s_%s_%s_%s_%s_%s.csv'
                                           %(vehName, vehLeader, round(traj_df[speed_spacing_column[0]][i], 1),
                                             traj_df['Time'][i], traj_df['Time'][i + x - 1], k))



def get_name(filename):
    return filename.split('.')[0]

def read_data_from_csv(folder_name):
    trajDict = {}
    vehicleInfoDict = {}
    statusDict = {}
    for csv_file in os.listdir(os.getcwd() + folder_name):
        if 'part' not in csv_file:
            continue
        infoRow = 4

        platoonName = get_name(csv_file)
        info = pd.read_csv(os.getcwd() + folder_name + csv_file, header=None, sep=',',engine='python',
                              names=np.arange(12), nrows=infoRow, index_col=False)
        numOfVehicle = int(info.iloc[2][1])
        vehicleOrder = info.iloc[1][1 : numOfVehicle + 1]
        accMode = info.iloc[3][1]

        headway = 'min'
        vehicleInfoDict[platoonName] = [numOfVehicle, vehicleOrder, accMode, headway]

        traj_df = pd.read_csv(os.getcwd() + folder_name + csv_file, header=infoRow, engine='python', sep=',')
        relevantCol = ['Time'] + [col for col in traj_df.columns if 'Speed' in col] + [col for col in traj_df.columns if 'IVS' in col]
        trajDict[platoonName] = traj_df[:][relevantCol].astype('float32')
        statusCol = ['Time'] + [col for col in traj_df.columns if 'Driver' in col]
        statusDict[platoonName] = traj_df[:][statusCol]
        statusDict[platoonName]['Time'] = statusDict[platoonName]['Time'].astype('float32')

    return trajDict, vehicleInfoDict, statusDict


def main():
    trajDict, vehicleInfoDict, statusDict = read_data_from_csv('/platooned_data/Vicolungo_data/')
    find_equilibrium(trajDict, vehicleInfoDict, statusDict)


if __name__=='__main__':
    main()
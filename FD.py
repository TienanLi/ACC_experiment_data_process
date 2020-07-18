import pickle
import os
import pandas as pd
from FD_functions import read_data_from_equlirbium_csv, find_equilibrium, read_data_from_summary_csv_overall, \
    get_headway_period, draw_FD, get_FD_parameters, s_v_figure, q_k_figure, exclude_outlier, read_disturbance, \
    disturbance_threshold


def main_multi_day(date_group):
    ACC_headway = {}
    headway_setting = ['3','1']
    headway_1_period, headway_3_period = get_headway_period()

    for headway in headway_setting:
        ACC1 = pd.DataFrame()
        ACC2 = pd.DataFrame()
        print('\nheadway',headway)
        for date in date_group:
            print(date)
            headway_period = {'1': headway_1_period[date], '3': headway_3_period[date]}
            if len(headway_period[headway]) == 0:
                continue
            traj_dict = read_data_from_summary_csv_overall('/platooned_data/03-%s-2020/'%(str(date).zfill(2)),
                                                           headway_period[headway], date)
            equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date, headway)

            # equilibrium_status_ACC1, equilibrium_status_ACC2 = \
            #     read_data_from_equlirbium_csv('/platooned_data/03-08-2020/equilibrium_traj/power',
            #                                                headway_period[headway])

            disturbance_start_end = read_disturbance('/platooned_data/03-%s-2020/'%(str(date).zfill(2)))
            equilibrium_status_ACC1 = disturbance_threshold(equilibrium_status_ACC1, disturbance_start_end)
            equilibrium_status_ACC2 = disturbance_threshold(equilibrium_status_ACC2, disturbance_start_end)

            ACC1 = pd.concat([ACC1, equilibrium_status_ACC1])
            ACC2 = pd.concat([ACC2, equilibrium_status_ACC2])
        filtered_ACC1 = exclude_outlier(ACC1[(ACC1[0] > 5) & (ACC1[0] < 70)].sort_values(by=[0]))
        print('ACC1 all:',len(filtered_ACC1))

        filtered_ACC2 = exclude_outlier(ACC2[(ACC2[0] > 5) & (ACC2[0] < 70)].sort_values(by=[0]))
        print('ACC2 all:',len(filtered_ACC2))
        ACC_headway[headway] = pd.concat([filtered_ACC1, filtered_ACC2])
        draw_FD(filtered_ACC1, filtered_ACC2, headway, '/platooned_data/03-08-2020/')

    fo = open(os.path.dirname(__file__) + '/platooned_data/03-08-2020/FD_data_08_normal', 'wb')
    pickle.dump(ACC_headway, fo)
    fo.close()
    print('-combined')
    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])
    s_v_figure(speed1,spacing1,speed2,spacing2,'headway 1','headway 3','/platooned_data/03-08-2020/','-combined')
    q_k_figure(density1, volumn1, density2, volumn2,'headway 1','headway 3','/platooned_data/03-08-2020/','-combined')

def main_single_day(folder_name, date):
    ACC_headway = {}
    headway_setting = ['3', '1']
    headway_1_period, headway_3_period = get_headway_period()
    headway_period = {'1': headway_1_period[date], '3': headway_3_period[date]}
    for headway in headway_setting:
        print('\nheadway', headway)
        if len(headway_period[headway]) == 0:
            continue



        # traj_dict = read_data_from_summary_csv_overall(folder_name, headway_period[headway], date)
        # equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date, headway,
        #                                                                     disturbance_info = disturbance_start_end)

        equilibrium_status_ACC1, equilibrium_status_ACC2 = \
            read_data_from_equlirbium_csv('/platooned_data/03-%s-2020/equilibrium_traj/' % (str(date).zfill(2)),
                                          headway_period[headway])

        disturbance_start_end = read_disturbance(folder_name)
        equilibrium_status_ACC1 = disturbance_threshold(equilibrium_status_ACC1, disturbance_start_end)
        equilibrium_status_ACC2 = disturbance_threshold(equilibrium_status_ACC2, disturbance_start_end)


        draw_FD(equilibrium_status_ACC1, equilibrium_status_ACC2, headway, folder_name)
        ACC_headway[headway] = pd.concat([equilibrium_status_ACC1, equilibrium_status_ACC2])

    fo = open(os.path.dirname(__file__) + '/platooned_data/03-%s-2020/equilibrium_traj/FD_data_%s'
              % (str(date).zfill(2), str(date).zfill(2)), 'wb')
    pickle.dump(ACC_headway, fo)
    fo.close()

    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])

    s_v_figure(speed1, spacing1, speed2, spacing2, 'headway 1', 'headway 3', folder_name, '-combined')
    q_k_figure(density1, volumn1, density2, volumn2, 'headway 1', 'headway 3', folder_name, '-combined')

if __name__=='__main__':
    Multi = True
    experiment_date = [7,8]#remember to change equlibrium save path at find_equlibrium function

    if Multi:
        main_multi_day(experiment_date)
    else:
        for date in experiment_date:
            print(date)
            traj_dict = main_single_day('/platooned_data/03-%s-2020/'%(str(date).zfill(2)), date)

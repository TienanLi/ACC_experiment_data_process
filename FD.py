import pickle
import os
import pandas as pd
from FD_functions import read_data_from_equlirbium_csv, find_equilibrium, read_data_from_summary_csv_overall, \
    get_headway_period, draw_FD, get_FD_parameters, s_v_figure, q_k_figure, exclude_outlier, read_disturbance, \
    disturbance_threshold

def main_multi_day_prius(date_group, mode):

    ACC_headway = {}
    headway_setting = ['3','1']
    headway_1_period, headway_3_period = get_headway_period(mode=mode)

    for headway in headway_setting:
        ACC1 = pd.DataFrame()
        ACC2 = pd.DataFrame()
        print('\nheadway',headway)
        for date in date_group:
            print(date)
            headway_period = {'1': headway_1_period[date], '3': headway_3_period[date]}
            if len(headway_period[headway]) == 0:
                continue
            # traj_dict = read_data_from_summary_csv_overall('/platooned_data/03-%s-2020/'%(str(date).zfill(2)),
            #                                                headway_period[headway], date)
            # equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date, headway)

            equilibrium_status_ACC1, equilibrium_status_ACC2 = \
                read_data_from_equlirbium_csv('/platooned_data/03-08-2020/equilibrium_traj/%s'%mode,
                                                           headway, exclude_outliers = False)

            disturbance_start_end = read_disturbance('/platooned_data/03-%s-2020/'%(str(date).zfill(2)))
            if len(equilibrium_status_ACC1) > 0:
                equilibrium_status_ACC1 = disturbance_threshold(equilibrium_status_ACC1, disturbance_start_end)
            if len(equilibrium_status_ACC2) > 0:
                equilibrium_status_ACC2 = disturbance_threshold(equilibrium_status_ACC2, disturbance_start_end,
                                                                left_bound=-5)

            ACC1 = pd.concat([ACC1, equilibrium_status_ACC1])
            if len(ACC1) > 0:
                ACC1 = ACC1.sort_values(by=[0])
            ACC2 = pd.concat([ACC2, equilibrium_status_ACC2])
            if len(ACC2) > 0:
                ACC2 = ACC2.sort_values(by=[0])

            break # if directly read equilibrium, 7 and 8 are already combined in one folder, no need to iterate dates

        # ACC1 = exclude_outlier(ACC1).sort_values([0])
        # ACC2 = exclude_outlier(ACC2).sort_values([0])

        ACC_headway[headway] = pd.concat([ACC1, ACC2])

        draw_FD(ACC1, ACC2, headway, '/platooned_data/03-08-2020/')

    fo = open(os.path.dirname(__file__) + '/platooned_data/03-08-2020/equilibrium_traj/%s/FD_data_08_%s'%(mode,mode),
              'wb')
    pickle.dump(ACC_headway, fo)
    fo.close()
    print('-combined')
    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])
    s_v_figure(speed1,spacing1,speed2,spacing2,'headway 1','headway 3','/platooned_data/03-08-2020/','-combined')
    q_k_figure(density1, volumn1, density2, volumn2,'headway 1','headway 3','/platooned_data/03-08-2020/','-combined')


def main_multi_day_tesla_x(date_group):
    ACC_headway = {}
    headway_setting = ['1','3']
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

            direct_read = True
            if direct_read:
                equilibrium_status_ACC1, equilibrium_status_ACC2 = \
                    read_data_from_equlirbium_csv('/platooned_data/02-23-2020/equilibrium_traj/',
                                                  headway, exclude_outliers=False, spacing_bound=60,max_speed=70)
            else:
                traj_dict = read_data_from_summary_csv_overall('/platooned_data/02-%s-2020/'%(str(date).zfill(2)),
                                                               headway_period[headway], date)
                equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date, headway,
                                                                                    spacing_upper_bound=60)

            ACC1 = pd.concat([ACC1, equilibrium_status_ACC1])
            ACC2 = pd.concat([ACC2, equilibrium_status_ACC2])

            if len(ACC1) > 0:
                ACC1 = ACC1.sort_values(by=[0])

            if len(ACC2) > 0:
                ACC2 = ACC2.sort_values(by=[0])

            if direct_read:
                break

        # ACC1 = exclude_outlier(ACC1).sort_values([0])
        # ACC2 = exclude_outlier(ACC2).sort_values([0])

        ACC_headway[headway] = pd.concat([ACC1, ACC2])
        draw_FD(ACC1, ACC2, headway, '/platooned_data/02-23-2020/')

    fo = open(os.path.dirname(__file__) + '/platooned_data/02-23-2020/FD_data_212223', 'wb')
    pickle.dump(ACC_headway, fo)
    fo.close()
    print('-combined')
    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])
    s_v_figure(speed1,spacing1,speed2,spacing2,'headway 1','headway 3','/platooned_data/02-23-2020/','-combined')
    q_k_figure(density1, volumn1, density2, volumn2,'headway 1','headway 3','/platooned_data/02-23-2020/','-combined')

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
        # equilibrium_status_ACC1, equilibrium_status_ACC2 = find_equilibrium(traj_dict, date, headway)

        equilibrium_status_ACC1, equilibrium_status_ACC2 = \
            read_data_from_equlirbium_csv('/platooned_data/03-%s-2020/equilibrium_traj/' % (str(date).zfill(2)),
                                          headway, exclude_outliers = False)

        disturbance_start_end = read_disturbance(folder_name)
        equilibrium_status_ACC1 = disturbance_threshold(equilibrium_status_ACC1, disturbance_start_end)
        equilibrium_status_ACC2 = disturbance_threshold(equilibrium_status_ACC2, disturbance_start_end)

        ACC_headway[headway] = pd.concat([equilibrium_status_ACC1, equilibrium_status_ACC2])
        draw_FD(equilibrium_status_ACC1, equilibrium_status_ACC2, headway, folder_name)

    fo = open(os.path.dirname(__file__) + '/platooned_data/03-%s-2020/equilibrium_traj/FD_data_%s'
              % (str(date).zfill(2), str(date).zfill(2)), 'wb')
    pickle.dump(ACC_headway, fo)
    fo.close()

    spacing1, speed1, density1, volumn1, spacing2, speed2, density2, volumn2 = \
        get_FD_parameters(ACC_headway['1'], ACC_headway['3'])

    s_v_figure(speed1, spacing1, speed2, spacing2, 'headway 1', 'headway 3', folder_name, '-combined')
    q_k_figure(density1, volumn1, density2, volumn2, 'headway 1', 'headway 3', folder_name, '-combined')

if __name__=='__main__':
    experiment_date = [12]#remember to change equlibrium save path at find_equlibrium function

    if (21 in experiment_date) and (experiment_date == [21,22,23]):
        main_multi_day_tesla_x(experiment_date)
        quit()
    if (7 in experiment_date) and (experiment_date == [7,8]):
        main_multi_day_prius(experiment_date, 'power')
        quit()
    else:
        for date in experiment_date:
            print(date)
            traj_dict = main_single_day('/platooned_data/03-%s-2020/'%(str(date).zfill(2)), date)

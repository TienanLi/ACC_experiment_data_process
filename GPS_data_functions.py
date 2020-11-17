# import folium
import os
import time
import pickle
import numpy as np
# import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.cm as pcm
# import geopy.distance as geod
from matplotlib.ticker import PercentFormatter
from matplotlib import rc
from math import ceil, floor
from Analyze_functions_multiple_veh import overlap_period
from base_functions import find_nearest_index, moving_average, get_a_part_before_a_point, \
    cal_ita, cal_ita_dynamic_wave_fix_tau0, min_speed_fine_measurment, min_speed_fine_measurment_2
from oscillation_functions import oscillation_statistics, save_oscillations
from matplotlib.collections import LineCollection

global data_frequency
data_frequency = 10
font = {'family': 'DejaVu Sans',
        'size': 16}
rc('font', **font)

def read_data_from_summary_csv(folder_name, platoon_number):
    traj_dict=[]
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if '%s.csv'%platoon_number not in csv_file:
            continue
        traj_dict.append([])
        fo = open(os.path.dirname(__file__)+folder_name+'/' + csv_file, 'r')
        line_num=0
        while True:
            line_num+=1
            line = fo.readline()
            if not line:
                break
            tmp = line.split(',')
            traj_dict[-1].append([float(tmp[i]) for i in range(len(tmp) - 1)])
        fo.close()

    output=[]
    for traj in traj_dict:
        output.append([[item[i] for item in traj] for i in range(len(traj[0]))])
    return output

def read_data_from_seperated_csv(file_name,Lat_column,Lon_column,speed_column,time_column,height_column):
    location = {}
    start_end_time = {}
    for veh in os.listdir(os.path.dirname(__file__)+file_name):
        if '.' in veh:
            continue
        if '_' in veh:
            continue
        location[veh] = []
        start_end_time[veh] = []
        for csv_file in os.listdir(os.path.dirname(__file__)+file_name+'/'+veh):
            if '.csv' not in csv_file:
                continue
            if '_' not in csv_file:
                continue
            fo = open(os.path.dirname(__file__)+file_name+'/'+veh + '/' + csv_file, 'r')
            line_num=0
            while True:
                line_num+=1
                line = fo.readline()
                if not line:
                    break
                tmp = line.split(',')
                if len(tmp) < max([Lat_column, Lon_column, speed_column, height_column]):
                    break
                # from m/s to mph
                location[veh].append([float(tmp[Lat_column]), float(tmp[Lon_column]),
                                      float(tmp[speed_column])*2.23694, round(float(tmp[time_column]),1), float(tmp[height_column])])

                if line_num == 1:
                    start_time = location[veh][-1][3]
            fo.close()
            start_end_time[veh].append((start_time, location[veh][-1][3]))

    return location, start_end_time

def map_visulization(location_sets, filename = None, l_num_max = None):
    l_num = 0
    file_num = 0
    colorscale = cm.LinearColormap(('r','y','g'),vmin=0,vmax=9)
    for location_index in range(min([len(x) for x in location_sets])):
        veh = 0
        for location in location_sets:
            l = location[location_index]
            if l_num == 0:
                m = folium.Map(location=[np.mean([l[0] for i in range(len(location_sets[0]))]),
                                         np.mean([l[1] for i in range(len(location_sets[0]))])],
                               zoom_start=13)
                file_num += 1
            folium.Circle(
                location=[l[0], l[1]],
                radius=1,
                color = colorscale(l_num%10),
                fill=True
            ).add_to(m)
            veh += 1
            l_num += 1
        if l_num_max is not None:
            if l_num > l_num_max:
                if filename is None:
                    m.save(str(file_num) +'test.html')
                else:
                    m.save(str(file_num) + filename)
                l_num = 0


def fix_missing_GPS_frame(t, param_set):
    new_t = []
    new_param = []
    param_num = len(param_set)
    for param in param_set:
        new_param.append([])
    t_range = range(int(t[0]*data_frequency), int(t[-1]*data_frequency) + 1)
    for tt in t_range:
        current_t = round(tt/data_frequency,1)
        new_t.append(current_t)
        if current_t in t:
            last_i = t.index(current_t)
            for p in range(param_num):
                new_param[p].append(param_set[p][last_i])
        else:
            for p in range(param_num):
                new_param[p].append(np.mean([param_set[p][last_i],param_set[p][last_i+1]]))

    return new_t, new_param


def traj_process(location, start_end_time, folder_name, platoon_number):
    divided_location = available_in_all(location, start_end_time, platoon_number)
    part = 1
    traj_dict=[]
    print('trajectory split get')
    for location in divided_location:
        try:
            file = open(os.path.dirname(__file__) + '/raw_data_split', 'rb')
            [traj, coordinate, heights] = pickle.load(file)
            file.close()
        except:
            traj = []
            coordinate = []
            heights =[]
            for veh in location:
                t = [location[veh][i][3] for i in range(len(location[veh]))]
                v = [location[veh][i][2] for i in range(len(location[veh]))] #mph
                lat = [location[veh][i][0] for i in range(len(location[veh]))]
                lon = [location[veh][i][1] for i in range(len(location[veh]))]
                height = [location[veh][i][4] for i in range(len(location[veh]))]
                t, [v, lat, lon, height] = fix_missing_GPS_frame(t, [v, lat, lon, height])
                print(veh+' trajectory fixed')
                if len(traj) == 0:
                    traj.append(t)
                    traj.append(v)
                else:
                    traj.append(v)
                coordinate.append(lat)
                coordinate.append(lon)
                heights.append(height)
        traj_with_coordinate = traj + coordinate
        traj_with_coordinate = [item[:min([len(item1) for item1 in traj])] for item in traj_with_coordinate]

        # the location of first veh - from speed
        distance = [[]]
        for tt in range(len(traj_with_coordinate[0])):
            if len(distance[0]) == 0:
                distance[0].append(0)
            else:
                distance[0].append(traj_with_coordinate[1][tt-1] * 0.44704 / data_frequency + distance[0][-1])
        print('veh1 traj get')
        #location of following vehs - from gap between GPS points
        total_veh_num = len(location)
        for veh in range(1, total_veh_num):

            front_veh_coordinate = [(traj_with_coordinate[veh * 2 + total_veh_num - 1][i],
                                     traj_with_coordinate[veh * 2 + total_veh_num][i])
             for i in range(len(traj_with_coordinate[0]))]

            my_distance, my_projection = distance_calculation_from_projection(
                front_veh_coordinate,
                [(traj_with_coordinate[veh * 2 + total_veh_num + 1][i],
                  traj_with_coordinate[veh * 2 + total_veh_num + 2][i])
                 for i in range(len(traj_with_coordinate[0]))],
                distance[veh - 1], traj_with_coordinate[veh + 1])
            distance.append(my_distance)
            print('veh'+str(veh+1)+' traj get')

        traj = traj + distance + heights
        traj = [item[:min([len(item1) for item1 in traj])] for item in traj]
        traj_dict.append(traj)
        save_traj(traj, folder_name, str(part)+'_'+str(platoon_number))
        part += 1
    return traj_dict

def distance_calculation_from_projection(front_car_coor, my_coor, front_car_location, my_speed):
    my_location = []
    my_projection = []
    consider_pre_time = 50 #data_points, corresponding to 5 sec
    for i in range(len(my_coor)):
        p_me = my_coor[i]
        p_target = front_car_coor[i]
        speed_me = my_speed[i] * 0.44704  # mph to m/s

        if (i - consider_pre_time < 0) or (i  > len(front_car_coor) - 2) or (speed_me < 2): #direct calculation
        # if True: #direct calculation
            my_location.append(front_car_location[i] - geod.distance(p_target, p_me).m)
            this_projection = p_me
        else: #using projection
            me_last_distance = my_location[i - 1]
            target_point_chain = front_car_coor[i - consider_pre_time: i + 2]
            target_point_chain_locations = front_car_location[i - consider_pre_time: i + 2]
            this_location, this_projection = projection(target_point_chain, target_point_chain_locations,
                                                    p_me, me_last_distance, speed_me)
            my_location.append(this_location)
        my_projection.append(this_projection)
    return my_location, my_projection

def projection(target_point_chain, target_point_chain_locations, p_me, me_last_location, speed_me):
    data_frequency = 10
    candidates_p_me_location = []
    candidates_p_me_project = {}
    for i in range(len(target_point_chain) - 1):
        traj_subsection = (target_point_chain[i], target_point_chain[i + 1])
        In_segment, p_me_project = check_projection(traj_subsection, p_me)
        if In_segment:
            if i < len(target_point_chain) - 2:
                gap = geod.distance(target_point_chain[i + 1], p_me_project).m
                my_location = target_point_chain_locations[i + 1] - gap
            else:#if p_me is somehow projected to the front of p_target
                p_target = target_point_chain[-2]
                gap = geod.distance(p_target, p_me_project).m
                if gap > 10:
                    print(gap)
                    gap = geod.distance(p_target, p_me).m
                    p_me_project = p_me
                my_location = target_point_chain_locations[-2] - gap
            candidates_p_me_location.append(my_location)
            candidates_p_me_project[my_location] = p_me_project

    if len(candidates_p_me_location) == 0:
        p_target = target_point_chain[-2]
        gap = geod.distance(p_target, p_me).m
        my_location = target_point_chain_locations[-2] - gap
        candidates_p_me_project[my_location] = p_me
    elif len(candidates_p_me_location) == 1:
        my_location = candidates_p_me_location[0]
    else:
        my_speed_location = me_last_location + speed_me / data_frequency
        my_location = candidates_p_me_location[np.argmin([abs(l-my_speed_location) for l in candidates_p_me_location])]
    return my_location, candidates_p_me_project[my_location]

def check_projection(segment, point):
    dx = segment[1][0] - segment[0][0]
    dy = segment[1][1] - segment[0][1]
    if dx == 0 and dy == 0:
        return False, False
    inner_product = (point[0] - segment[0][0]) * dx + (point[1] - segment[0][1]) * dy
    if (inner_product >= 0) and (inner_product <= (dx * dx + dy * dy)):
        projected_length = inner_product / (dx * dx + dy * dy)
        projected_point = (segment[0][0] + projected_length * dx,
                           segment[0][1] + projected_length * dy)
        return True, projected_point
    else:
        return False, False

def save_continuous_CF_traj(traj_dict, oscillation_set, folder_name):
    #need to distinguish different headway vehicle mode
    oscillation_set.sort(key=lambda o: o[0])
    continuous_part = []
    start = np.nan
    for i in range(1, len(oscillation_set)):
        if np.isnan(start):
            start = oscillation_set[i-1][0]
        if (oscillation_set[i][0] - oscillation_set[i-1][1] >= 20) or (i == len(oscillation_set)):
            continuous_part.append(
                (start, oscillation_set[i-1][1], oscillation_set[i-1][2], oscillation_set[i-1][3]))
            start = np.nan
            continue
        if (oscillation_set[i][2] != oscillation_set[i - 1][2]) or \
                (oscillation_set[i][3] != oscillation_set[i - 1][3]):
            continuous_part.append(
                (start, oscillation_set[i - 1][1], oscillation_set[i - 1][2], oscillation_set[i - 1][3]))
            start = np.nan


    part = 0
    for traj in traj_dict:
        divided_traj, selected_oscillation_set = traj_by_oscillation_manual(traj, continuous_part)
        for i in range(len(divided_traj)):
            d_traj = divided_traj[i]
            save_traj(d_traj, folder_name + 'split_traj/', 'continuous_part_' + str(part) +
                      '_headway_' + selected_oscillation_set[i][2] + '_mode_' + selected_oscillation_set[i][3])
            part += 1

def get_tau0_congested_s(date, headway, engine_mode):
    # Tesla
    if date == '12':
        if headway == '3':
            tau0 = 0.89
            congested_s = 15.97
        else:
            tau0 = 0.48
            congested_s = 16.66
    if date == '14':
        if headway == '3':
            tau0 = 2.06
            congested_s = 5.5
        else:
            tau0 = 1.07
            congested_s = 5.41
    if date == '15':
        if headway == '3':
            tau0 = 2.06
            congested_s = 5.5
        else:
            tau0 = 1.07
            congested_s = 5.41


    if date in ['07','08']:
        if engine_mode == 'Normal':
            if headway == '3':
                tau0 = 2.18
                congested_s = 13.15
            else:
                tau0 = 0.85
                congested_s = 12.37
        else:
            if headway == '3':
                tau0 = 2.33
                congested_s = 13.15
            else:
                tau0 = 0.85
                congested_s = 12.37

    return tau0, congested_s

def speed_visulization_2_subplot(traj_dict, folder_name, first_fig, MA_window = 2, extended_period = 50, draw_veh=3):
    #'oblique_traj', 'traj', 'spacing', 'eta'
    HIGH_PLOT = False

    X_LIMIT = [-18, 15]
    traj_color = ['green', 'red', 'blue']
    traj_label = ['Leader-HDV', 'ACC1', 'ACC2']


    for traj in traj_dict:

        if MA_window > 0:
            for i in range(1, len(traj)):
                traj[i] = moving_average(traj[i], MA_window * data_frequency)

        oscillation_set = read_oscillation_time(folder_name)
        divided_traj, selected_oscillation_set = traj_by_oscillation_manual(traj, oscillation_set)
        traj_num = 0
        overall_traj = traj.copy()
        for traj in divided_traj:
            # which figure to produce
            # DRAW = False
            # if selected_oscillation_set[traj_num][-1] in [52]:
            # #     if selected_oscillation_set[traj_num][2] == '1':
            # #         if selected_oscillation_set[traj_num][4] == 'mild':
            # # if selected_oscillation_set[traj_num][6].replace('\n','') == 'long':
            # #     #             if selected_oscillation_set[traj_num][5] == 'mild':
            #     DRAW = True
            # if not DRAW:
            #     traj_num += 1
            #     continue



            oscillations_LV, WT_frequncy = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
            if len(oscillations_LV[0]) > 0:
                mid_point = oscillations_LV[0][0]
            else:
                print(selected_oscillation_set[traj_num][-1], ': can\'t find oscillation')
                traj_num += 1
                continue

            extended_traj, non_used_value = traj_by_oscillation_manual(overall_traj, [(mid_point - extended_period,
                                                                                       mid_point + extended_period)])

            if selected_oscillation_set[traj_num][6].replace('\n', '') == 'long':
                cruise = True
            else:
                cruise = False

            extended_traj = extended_traj[0]
            traj[0] = [tt - mid_point for tt in traj[0]]
            extended_traj[0] = [tt - mid_point for tt in extended_traj[0]]

            oscillations_LV, WT_frequncy = oscillation_statistics(traj[0], traj[1], data_frequency,
                                                                  fluent=True, cruise=cruise)
            try:
                oscillations_FV, WT_frequncy = oscillation_statistics(traj[0], traj[2], data_frequency,
                                                                      fluent=True, cruise=cruise,
                                                                      end_speed_bound=oscillations_LV[0][5])
            except:
                oscillations_FV = [[]]
                print(selected_oscillation_set[traj_num][-1], 'ACC1 oscillation not recognized')
            if len(traj) > 3:
                try:
                    oscillations_TV, WT_frequncy = oscillation_statistics(traj[0], traj[3], data_frequency,
                                                                          fluent=True, cruise=cruise,
                                                                          end_speed_bound=oscillations_FV[0][5])
                except:
                    oscillations_TV = [[]]
                    print(selected_oscillation_set[traj_num][-1], 'ACC2 oscillation not recognized')



            y_limit = [[20, 40], [30, 50], [50, 70]]
            if oscillations_LV[0][3] > 55:
                used_y_limit = y_limit[2]
                speed_group = 'high'
            elif oscillations_LV[0][3] > 40:
                used_y_limit = y_limit[1]
                speed_group = 'mid'
            else:
                used_y_limit = y_limit[0]
                speed_group = 'low'

            if extended_period is not None:
                width = 12
            else:
                width = 5
            if HIGH_PLOT:
                width=9
                label_size = 16
            else:
                label_size =24

            spacing_FV = [extended_traj[4][j] - extended_traj[5][j] for j in range(len(extended_traj[5]))]
            spacing_TV = [extended_traj[5][j] - extended_traj[6][j] for j in range(len(extended_traj[6]))]

            t = extended_traj[0]

            d_HV = extended_traj[4]
            d_ACC1 = extended_traj[5]
            d_ACC2 = extended_traj[6]

            headway = selected_oscillation_set[traj_num][2]
            engine_mode = selected_oscillation_set[traj_num][3]
            date = folder_name[-8:-6]
            tau0, congested_s = get_tau0_congested_s(date, headway, engine_mode)

            t_ita_ACC1, ita_ACC1 = cal_ita_dynamic_wave_fix_tau0(t, d_HV, t, d_ACC1, tau0, congested_s)
            t_ita_ACC1_fix_wave, ita_ACC1_fix_wave = cal_ita(t, d_HV, t, d_ACC1, sim_freq=0.1, w=congested_s / tau0, k=1 / congested_s)

            t_ita_ACC2, ita_ACC2 = cal_ita_dynamic_wave_fix_tau0(t, d_ACC1, t, d_ACC2, tau0, congested_s)

            fine_measurement_list = [76, 77, 80, 91, 96, 48]
            fine_measurement_list2 = [30, 25]
            if selected_oscillation_set[traj_num][-1] in fine_measurement_list:
                oscillations_LV, oscillations_FV = min_speed_fine_measurment(oscillations_LV, oscillations_FV,
                                                                             extended_traj[1], extended_traj[2],
                                                                             extended_traj[0])
            if selected_oscillation_set[traj_num][-1] in fine_measurement_list2:
                oscillations_LV = min_speed_fine_measurment_2(oscillations_LV, oscillations_FV,
                                                               extended_traj[1],
                                                               extended_traj[0])





            fig = plt.figure(figsize=(width, 10), dpi=100)
            bx = fig.add_subplot(211)
            if HIGH_PLOT:
                bx.set_position([0.15, 0.4, 0.8, 0.55])
            else:
                bx.set_position([0.1, 0.5, 0.87, 0.35])

            slope = (extended_traj[4][-1] - extended_traj[4][0]) / (len(extended_traj[4]) - 1) -1

            plt.xlim(X_LIMIT)
            plt.grid(True)
            plt.title('OSCILLATION: ' + str(selected_oscillation_set[traj_num][-1]) +
                      ', HEADWAY: ' + selected_oscillation_set[traj_num][2] +
                      # ', ENGINE MODE: ' + selected_oscillation_set[traj_num][3] +
                      # '\nMAGNITUDE: ' + selected_oscillation_set[traj_num][4] +
                      # ', BRAKE PATTERN: ' + selected_oscillation_set[traj_num][5] +
                      ', CRUISE: ' + selected_oscillation_set[traj_num][6].replace('\n', '') + '\n',fontsize=12)
            # ', oblique v_0:' + str(round(slope * 10 / 0.44704, 2)) + 'mph\n')

            if first_fig == 'spacing':
                plt.plot(extended_traj[0], spacing_FV,
                         'r', label='spacing_veh1-2', linewidth=2)

                plt.ylabel('spacing (m)', fontsize=label_size)

            if first_fig in ['oblique_traj', 'traj']:

                max_position = 0
                min_position = 0

                for i in range(1, draw_veh + 1):
                    # oblique
                    if first_fig == 'oblique_traj':
                        oblique_traj = [extended_traj[i + 3][j] - extended_traj[6][0] - slope * j
                                        for j in range(len(extended_traj[i + 3]))]
                        max_position = max(max(oblique_traj), max_position)
                        min_position = min(min(oblique_traj), min_position)
                        plt.ylabel('oblique location(m)', fontsize=label_size)
                        start_points = []
                        end_points = []

                        for o in oscillations_LV:
                            bx.scatter(o[2], extended_traj[4][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10, color='g', s=60,zorder=10)
                            bx.scatter(o[6], extended_traj[4][find_nearest_index(extended_traj[0], o[6])] - extended_traj[6][0] - slope * (o[6]+65)*10, color='g', s=60,zorder=10)
                            bx.scatter(o[8], extended_traj[4][find_nearest_index(extended_traj[0], o[8])] - extended_traj[6][0] - slope * (o[8]+65)*10, color='g', s=60,zorder=10)
                            bx.scatter(o[4], extended_traj[4][find_nearest_index(extended_traj[0], o[4])] - extended_traj[6][0] - slope * (o[4]+65)*10, color='g', s=60,zorder=10)
                            start_points.append(extended_traj[4][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10)
                            end_points.append(extended_traj[4][find_nearest_index(extended_traj[0], o[4])] - extended_traj[6][0] - slope * (o[4]+65)*10)

                        for o in oscillations_FV:
                            if len(o) == 0:
                                continue
                            bx.scatter(o[2], extended_traj[5][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10, color='r', s=60,zorder=10)
                            bx.scatter(o[6], extended_traj[5][find_nearest_index(extended_traj[0], o[6])] - extended_traj[6][0] - slope * (o[6]+65)*10, color='r', s=60,zorder=10)
                            bx.scatter(o[8], extended_traj[5][find_nearest_index(extended_traj[0], o[8])] - extended_traj[6][0] - slope * (o[8]+65)*10, color='r', s=60,zorder=10)
                            bx.scatter(o[4], extended_traj[5][find_nearest_index(extended_traj[0], o[4])] - extended_traj[6][0] - slope * (o[4]+65)*10, color='r', s=60,zorder=10)
                            start_points.append(extended_traj[5][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10)
                            end_points.append(extended_traj[5][find_nearest_index(extended_traj[0], o[4])] - extended_traj[6][0] - slope * (o[4]+65)*10)

                        if draw_veh == 3:
                            for o in oscillations_TV:
                                if len(o) == 0:
                                    continue
                                bx.scatter(o[2], extended_traj[6][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10, color='b',
                                           s=60,zorder=10)
                                bx.scatter(o[6], extended_traj[6][find_nearest_index(extended_traj[0], o[6])] - extended_traj[6][0] - slope * (o[6]+65)*10, color='b',
                                           s=60,zorder=10)
                                bx.scatter(o[8], extended_traj[6][find_nearest_index(extended_traj[0], o[8])] - extended_traj[6][0] - slope * (o[8]+65)*10, color='b',
                                           s=60,zorder=10)
                                bx.scatter(o[4], extended_traj[6][find_nearest_index(extended_traj[0], o[4])]- extended_traj[6][0] - slope * (o[4]+65)*10, color='b',
                                           s=60,zorder=10)
                                start_points.append(extended_traj[6][find_nearest_index(extended_traj[0], o[2])] - extended_traj[6][0] - slope * (o[2]+65)*10)
                                end_points.append(extended_traj[6][find_nearest_index(extended_traj[0], o[4])]- extended_traj[6][0] - slope * (o[4]+65)*10)

                        if i != draw_veh:

                            newell_traj = [dd - congested_s - tau0 * 10 * slope for dd in oblique_traj]
                            plt.plot([tt + tau0 for tt in extended_traj[0]], newell_traj,
                                     'k--', label='Newell', linewidth=2, alpha = 1)

                            # for o in oscillations_LV:
                            #     bx.scatter(o[2]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[2])],
                            #                color='b', s=60, zorder=10,alpha=1)
                            #     bx.scatter(o[6]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[6])],
                            #                color='b', s=60, zorder=10,alpha=1)
                            #     bx.scatter(o[8]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[8])],
                            #                color='b', s=60, zorder=10,alpha=1)
                            #     bx.scatter(o[4]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[4])],
                            #                color='b', s=60, zorder=10,alpha=1)
                        max_position = max(end_points)
                        min_position = min(start_points)
                        plt.ylim([min_position - 10, max_position + 10])

                    # no oblique
                    if first_fig == 'traj':
                        oblique_traj = [extended_traj[i + 3][j]
                                        for j in range(len(extended_traj[i + 3]))]
                        start_points = []
                        end_points = []

                        if i != draw_veh:
                            newell_traj = [dd - congested_s for dd in oblique_traj]
                            plt.plot([tt + tau0 for tt in extended_traj[0]], newell_traj,
                                     'k--', label='Newell', linewidth=1, alpha = .5)
                            for o in oscillations_LV:
                                bx.scatter(o[2]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[2])],
                                           color='b', s=60, zorder=10,alpha=1)
                                bx.scatter(o[6]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[6])],
                                           color='b', s=60, zorder=10,alpha=1)
                                bx.scatter(o[8]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[8])],
                                           color='b', s=60, zorder=10,alpha=1)
                                bx.scatter(o[4]+tau0, newell_traj[find_nearest_index(extended_traj[0], o[4])],
                                           color='b', s=60, zorder=10,alpha=1)

                        for o in oscillations_LV:
                            bx.scatter(o[2], extended_traj[4][find_nearest_index(extended_traj[0], o[2])], color='g', s=60)
                            bx.scatter(o[6], extended_traj[4][find_nearest_index(extended_traj[0], o[6])], color='g', s=60)
                            bx.scatter(o[8], extended_traj[4][find_nearest_index(extended_traj[0], o[8])], color='g', s=60)
                            bx.scatter(o[4], extended_traj[4][find_nearest_index(extended_traj[0], o[4])], color='g', s=60)
                            start_points.append(extended_traj[4][find_nearest_index(extended_traj[0], o[2])])
                            end_points.append(extended_traj[4][find_nearest_index(extended_traj[0], o[4])])

                        for o in oscillations_FV:
                            if len(o) == 0:
                                continue
                            bx.scatter(o[2], extended_traj[5][find_nearest_index(extended_traj[0], o[2])], color='r', s=60)
                            bx.scatter(o[6], extended_traj[5][find_nearest_index(extended_traj[0], o[6])], color='r', s=60)
                            bx.scatter(o[8], extended_traj[5][find_nearest_index(extended_traj[0], o[8])], color='r', s=60)
                            bx.scatter(o[4], extended_traj[5][find_nearest_index(extended_traj[0], o[4])], color='r', s=60)
                            start_points.append(extended_traj[5][find_nearest_index(extended_traj[0], o[2])])
                            end_points.append(extended_traj[5][find_nearest_index(extended_traj[0], o[4])])
                        if draw_veh == 3:
                            for o in oscillations_TV:
                                if len(o) == 0:
                                    continue
                                bx.scatter(o[2], extended_traj[6][find_nearest_index(extended_traj[0], o[2])], color='b',
                                           s=60)
                                bx.scatter(o[6], extended_traj[6][find_nearest_index(extended_traj[0], o[6])], color='b',
                                           s=60)
                                bx.scatter(o[8], extended_traj[6][find_nearest_index(extended_traj[0], o[8])], color='b',
                                           s=60)
                                bx.scatter(o[4], extended_traj[6][find_nearest_index(extended_traj[0], o[4])], color='b',
                                           s=60)
                                start_points.append(extended_traj[6][find_nearest_index(extended_traj[0], o[2])])
                                end_points.append(extended_traj[6][find_nearest_index(extended_traj[0], o[4])])

                        plt.ylabel('location(m)', fontsize=label_size)
                        max_position = max(end_points)
                        min_position = min(start_points)
                        plt.ylim([min_position - 50, max_position + 50])


                    # plt.plot(np.array(extended_traj[0]), np.array(oblique_traj),label=traj_label[i-1])
                    # plt.legend()

                    color_indicator = np.array(extended_traj[i])
                    points = np.array([np.array(extended_traj[0]), np.array(oblique_traj)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(used_y_limit[0] + 5, used_y_limit[0] + 15 + 2.5)
                    lc = LineCollection(segments, cmap='jet_r', norm=norm)
                    lc.set_array(color_indicator)
                    lc.set_linewidth(2)#original - 4,
                    bx.add_collection(lc)

                # cmap_jet = pcm.get_cmap('jet_r')
                # sm = plt.cm.ScalarMappable(cmap=cmap_jet,
                #                            norm=plt.Normalize(vmin=used_y_limit[0] + 5, vmax=used_y_limit[0] + 15))
                # cbar = plt.colorbar(sm, orientation='horizontal', cax=plt.axes([0.2, 0.595, 0.65, 0.025]))
                # # cbar.set_label('speed (mph)', fontsize=16)
                # cbar.set_ticks([used_y_limit[0] + 5, used_y_limit[0] + 10, used_y_limit[0] + 15])

            if first_fig == 'eta':

                plt.ylim([.8, 1.2])

                # bx.plot(t_ita_ACC1, ita_ACC1, c=traj_color[1], linewidth=1 , label='d ratio')

                bx.plot(t_ita_ACC1_fix_wave, ita_ACC1_fix_wave, c='k', linewidth=1, label='t ratio (with interplot)')
                # plt.legend()

                # if draw_veh == 3:
                    # bx.plot(t_ita_ACC2, ita_ACC2, c=traj_color[2],  linewidth=1 , label='ACC2')
                    # plt.legend(loc=1)
                plt.ylabel('$\\eta$', fontsize=label_size)
                plt.xlim(X_LIMIT)
                plt.grid(True)
            ax = fig.add_subplot(212)
            if HIGH_PLOT:
                ax.set_position([0.15, 0.1, 0.8, 0.25])
            else:
                ax.set_position([0.1, 0.1, 0.87, 0.35])

            for i in range(1, draw_veh + 1):
                if i < draw_veh:
                    ax.plot([tt + tau0 for tt in extended_traj[0]], extended_traj[i], c=traj_color[i - 1],
                            linestyle='--',linewidth=2
                            ,label=traj_label[i - 1]+' (shifted by $\\tau_0$)'
                            )
                ax.plot(extended_traj[0], extended_traj[i], c=traj_color[i - 1], label=traj_label[i - 1])

            try:

                for o in oscillations_LV:
                    # ax.scatter(o[2] + tau0, o[3], color='b', s=60, alpha =1)
                    # ax.scatter(o[6] + tau0, o[7], color='b', s=60, alpha =1)
                    # ax.scatter(o[8] + tau0, o[9], color='b', s=60, alpha =1)
                    # ax.scatter(o[4] + tau0, o[5], color='b', s=60, alpha =1)
                    ax.scatter(o[2], o[3], color='g', s=60)
                    ax.scatter(o[6], o[7], color='g', s=60)
                    ax.scatter(o[8], o[9], color='g', s=60)
                    ax.scatter(o[4], o[5], color='g', s=60)
                    # ax.scatter(o[0], o[1], color='k', marker='*', s=120)
                for o in oscillations_FV:
                    if len(o) == 0:
                        continue
                    ax.scatter(o[2], o[3], color='r', s=60)
                    ax.scatter(o[6], o[7], color='r', s=60)
                    ax.scatter(o[8], o[9], color='r', s=60)
                    ax.scatter(o[4], o[5], color='r', s=60)
                    # ax.scatter(o[0], o[1], color='k', marker='*', s=120)
                for o in oscillations_TV:
                    if len(o) == 0:
                        continue
                    if o[2] > oscillations_FV[0][4]:
                        continue
                    if draw_veh < 3:
                        continue
                    ax.scatter(o[2], o[3], color='b', s=60)
                    ax.scatter(o[6], o[7], color='b', s=60)
                    ax.scatter(o[8], o[9], color='b', s=60)
                    ax.scatter(o[4], o[5], color='b', s=60)
                    # ax.scatter(o[0], o[1], color='k', marker='*', s=36)
            except:
                pass
            plt.xlim(X_LIMIT)
            ticks_in_ms = np.arange(round(used_y_limit[0] * 0.44704), round(used_y_limit[1] * 0.44704) + 2, 2)
            plt.yticks(ticks_in_ms * 2.23694, ticks_in_ms)
            plt.ylim([used_y_limit[0] - 4, used_y_limit[1]] ) #mild - 7, strong - 1
            plt.ylabel('speed (m/s)', fontsize=label_size)
            plt.grid(True)
            if not HIGH_PLOT:
                plt.legend(loc=4, fontsize=16)
            else:
                plt.legend(loc=4, fontsize=12)
            plt.xlabel('time (s)', fontsize=label_size)

            try:
                os.stat(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            except:
                os.mkdir(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            plt.savefig(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/split_' +
                        str(selected_oscillation_set[traj_num][-1]) + '.png')
            plt.close()

            traj_num += 1


def save_info(traj_dict, folder_name, MA_window = 2, extended_period = 50, overall=False, draw_veh=3):
    # save_continuous_CF_traj(traj_dict, oscillation_set, folder_name)

    oscillation_info = []
    complete_oscillation_info = []
    for traj in traj_dict:
        if MA_window > 0:
            for i in range(1, len(traj)):
                traj[i] = moving_average(traj[i], MA_window * data_frequency)
        if overall == True:
            v_diff = [traj[1][i] - traj[2][i] for i in range(len(traj[0])) if traj[2][i] > 5]
            d_diff = [traj[4][i] - traj[5][i] for i in range(len(traj[0])) if traj[2][i] > 5]
            e_diff = [traj[7][i] - traj[8][i] for i in range(len(traj[0])) if traj[2][i] > 5]
            fig = plt.figure(figsize=(7, 4), dpi=100)
            plt.hist(v_diff, bins = np.arange(-1,1.1,.05), weights=np.ones(len(v_diff)) / len(v_diff))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.title('speed error (mph)')
            plt.savefig('v_diff.png')
            plt.close()
            fig = plt.figure(figsize=(7, 4), dpi=100)
            plt.hist(d_diff, bins = np.arange(0, 8, .5), weights=np.ones(len(d_diff)) / len(d_diff))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.title('location error (m)')
            plt.savefig('d_diff.png')
            plt.close()
            fig = plt.figure(figsize=(7, 4), dpi=100)
            plt.hist(e_diff, weights=np.ones(len(e_diff)) / len(e_diff))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.title('elevation error (m)')
            plt.savefig('e_diff.png')
            plt.close()

            Time_range = np.arange(ceil(min(traj[0]) / 60) * 60, max(traj[0]), 30)
            fig = plt.figure(figsize=(7, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_position([0.08, 0.2, 0.82, 0.7])
            plt.plot(traj[0], [traj[1][i] - traj[2][i] for i in range(len(traj[0]))], label='speed diff (mph)')
            plt.plot(traj[0], [traj[4][i] - traj[5][i] for i in range(len(traj[0]))], label='location diff (m)')
            plt.plot(traj[0], [traj[7][i] - traj[8][i] for i in range(len(traj[0]))], label='elevation diff (m)')
            plt.legend()
            plt.xticks(Time_range)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('speed(mph)', fontsize=20)
            plt.grid(True)
            plt.show()

            continue
        oscillation_set = read_oscillation_time(folder_name)
        divided_traj, selected_oscillation_set = traj_by_oscillation_manual(traj, oscillation_set)
        traj_num = 0
        overall_traj = traj.copy()
        for traj in divided_traj:
            oscillations_LV, WT_frequncy = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
            if len(oscillations_LV[0]) > 0:
                mid_point = oscillations_LV[0][0]
            else:
                print(selected_oscillation_set[traj_num][-1], ': can\'t find oscillation')
                traj_num += 1
                continue

            extended_traj, non_used_value = traj_by_oscillation_manual(overall_traj, [(mid_point-extended_period,
                                                               mid_point+extended_period)])

            if selected_oscillation_set[traj_num][6].replace('\n', '') == 'long':
                cruise = True
            else:
                cruise = False

            extended_traj = extended_traj[0]
            traj[0] = [tt - mid_point for tt in traj[0]]
            extended_traj[0] = [tt - mid_point for tt in extended_traj[0]]

            oscillations_LV, WT_frequncy = oscillation_statistics(traj[0], traj[1], data_frequency,
                                                                  fluent=True, cruise=cruise)
            try:
                oscillations_FV, WT_frequncy = oscillation_statistics(traj[0], traj[2], data_frequency,
                                                                  fluent=True, cruise=cruise,
                                                                  end_speed_bound=oscillations_LV[0][5])
            except:
                oscillations_FV = [[]]
                print(selected_oscillation_set[traj_num][-1],'ACC1 oscillation not recognized')
            if len(traj) > 3:
                try:
                    oscillations_TV, WT_frequncy = oscillation_statistics(traj[0], traj[3], data_frequency,
                                                                          fluent=True, cruise=cruise,
                                                                          end_speed_bound=oscillations_FV[0][5])
                except:
                    oscillations_TV = [[]]
                    print(selected_oscillation_set[traj_num][-1],'ACC2 oscillation not recognized')
            y_limit = [[20, 40], [30, 50], [50, 70]]
            if oscillations_LV[0][3] > 55:
                used_y_limit = y_limit[2]
                speed_group = 'high'
            elif oscillations_LV[0][3] > 40:
                used_y_limit = y_limit[1]
                speed_group = 'mid'
            else:
                used_y_limit = y_limit[0]
                speed_group = 'low'

            if extended_period is not None:
                width = 12
            else:
                width = 7
            stablization_before_LV = get_a_part_before_a_point(extended_traj[1], extended_traj[0],
                                                               oscillations_LV[0][2], 30)
            stablization_before_FV = get_a_part_before_a_point(extended_traj[2], extended_traj[0],
                                                               oscillations_LV[0][2], 30)

            spacing_FV = [extended_traj[4][j] - extended_traj[5][j] for j in range(len(extended_traj[5]))]
            spacing_TV = [extended_traj[5][j] - extended_traj[6][j] for j in range(len(extended_traj[6]))]
            stablization_before_spacing_FV = get_a_part_before_a_point(spacing_FV, extended_traj[0],
                                                               oscillations_LV[0][2], 30)

            speed_diff_FV = [extended_traj[1][i] - extended_traj[2][i] for i in range(len(extended_traj[1]))]
            speed_diff_TV = [extended_traj[2][i] - extended_traj[3][i] for i in range(len(extended_traj[2]))]
            spacing_from_speed_FV = [sum(speed_diff_FV[:i + 1]) * 0.44704 / 10 for i in range(len(speed_diff_FV))]
            spacing_from_speed_TV = [sum(speed_diff_TV[:i + 1]) * 0.44704 / 10 for i in range(len(speed_diff_TV))]

            fine_measurement_list = [76, 77, 80, 91, 96, 48]
            fine_measurement_list2 = [30, 25]
            if selected_oscillation_set[traj_num][-1] in fine_measurement_list:
                oscillations_LV, oscillations_FV = min_speed_fine_measurment(oscillations_LV, oscillations_FV,
                                                                             extended_traj[1], extended_traj[2],
                                                                             extended_traj[0])
            if selected_oscillation_set[traj_num][-1] in fine_measurement_list2:
                oscillations_LV = min_speed_fine_measurment_2(oscillations_LV, oscillations_FV,
                                                               extended_traj[1],
                                                               extended_traj[0])
            oscillation_info.append((oscillations_LV[0],oscillations_FV[0],
                                     oscillations_TV[0],selected_oscillation_set[traj_num][2:]+[speed_group],
                                     stablization_before_LV, stablization_before_FV, stablization_before_spacing_FV,
                                     spacing_FV,spacing_from_speed_FV, extended_traj[1]))

            #full_speed, location
            LV_complete = (extended_traj[1])
            #full_speed, full_spacing, full_spacing_from_v, location
            FV_complete = (extended_traj[2], spacing_FV, spacing_from_speed_FV)
            #full_speed, full_spacing, full_spacing_from_v, location
            TV_complete = (extended_traj[3], spacing_TV, spacing_from_speed_TV)

            LOCATION_INFO = (extended_traj[4], extended_traj[5], extended_traj[6])

            complete_oscillation_info.append((oscillations_LV[0],oscillations_FV[0],oscillations_TV[0],
                                              selected_oscillation_set[traj_num][2:]+[speed_group],
                                              LV_complete, FV_complete, TV_complete, extended_traj[0],LOCATION_INFO,
                                              mid_point))

            traj_num += 1

    file = open(os.path.dirname(__file__) + folder_name + 'oscillation_info','wb')
    pickle.dump(oscillation_info, file)
    file.close()
    file = open(os.path.dirname(__file__) + folder_name + 'complete_oscillation_info','wb')
    pickle.dump(complete_oscillation_info, file)
    file.close()


def available_in_all(location, start_end_time, veh_num):
    i = 0
    time_of_considered_veh = {}
    for veh in location:
        i += 1
        if i > veh_num:
            break
        if i == 1:
            shared_period = start_end_time[veh]
        else:
            shared_period = overlap_period(shared_period, start_end_time[veh])
        time_of_considered_veh[veh] = [point[3] for point in location[veh]]
    considered_location = []
    for period in shared_period:
        considered_location.append({})
        for veh in time_of_considered_veh:
            start_index = find_nearest_index(time_of_considered_veh[veh],period[0])
            end_index = find_nearest_index(time_of_considered_veh[veh], period[1])
            considered_location[-1][veh] = location[veh][start_index:end_index]
    return considered_location

def save_traj(traj, folder_name, part):
        flink = open(os.path.dirname(__file__) + folder_name + 'summary_output_%s.csv'%part,'w')
        for i in range(len(traj[0])):
            for item in traj:
                flink.write('%s,'%item[i])
            flink.write('\n')
        flink.close()

def read_oscillation_time(folder_name):
    oscillation_set = []
    fo = open(os.path.dirname(__file__) + folder_name + '/oscillation_time.csv', 'r')
    fo.readline()
    line_num=0
    while True:
        line_num+=1
        line = fo.readline()
        if not line:
            break
        tmp = line.split(',')
        try:
            oscillation_set.append([int(tmp[0]), int(tmp[1]), tmp[6], tmp[8], tmp[9], tmp[10], tmp[11], line_num])
            #start, end, headway, power mode, magnitude, brake pattern, cruise, oscillation_num
        except:
            print('missing one', tmp[8], tmp[9], tmp[10], tmp[11])
    fo.close()
    return oscillation_set

def traj_by_oscillation_manual(traj, oscillation_set):
    divided_traj = []
    selected_oscillation_set = []
    for oscillation in oscillation_set:
        if (oscillation[0] > traj[0][-1]) or (oscillation[1] < traj[0][0]):
            continue
        s=find_nearest_index(traj[0],max(traj[0][0],oscillation[0]))
        e=find_nearest_index(traj[0],min(traj[0][-1],oscillation[1]))
        divided_traj.append([i[s:e] for i in traj])
        selected_oscillation_set.append(oscillation)
    return divided_traj, selected_oscillation_set


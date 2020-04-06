# import folium
import os
import time
import pickle
import numpy as np
# import branca.colormap as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.cm as pcm
import geopy.distance as geod

from matplotlib import rc
from math import ceil, floor
from Analyze_functions_multiple_veh import overlap_period
from base_functions import find_nearest_index, moving_average, cal_distance
from oscillation_functions import oscillation_statistics, traj_by_oscillation, save_oscillations
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
                                      float(tmp[speed_column])*2.23694, float(tmp[time_column]), float(tmp[height_column])])

                if line_num == 1:
                    start_time = location[veh][-1][3]
            fo.close()
            start_end_time[veh].append((start_time, location[veh][-1][3]))
    return location, start_end_time

def map_visulization(location):
    m = folium.Map(location=[np.mean([location[i][0] for i in range(len(location))]), np.mean([location[i][1] for i in range(len(location))])],zoom_start=13)
    colorscale = cm.LinearColormap(('r','y','g'),vmin=0,vmax=20)
    for l in location:
        folium.Circle(
            location=[l[0], l[1]],
            radius=1,
            color = colorscale(l[2]),
            fill=True
        ).add_to(m)
    colorscale.caption = 'Speed (m/s)'
    m.add_child(colorscale)
    m.save('test.html')

def fix_missing_GPS_frame(t, param_set):
    new_t = []
    new_param = []
    param_num = len(param_set)
    for param in param_set:
        new_param.append([])
    t_range = range(int(t[0]*data_frequency), int(t[-1]*data_frequency) + 1)
    i = 0
    for tt in t_range:
        current_t = round(tt/data_frequency,1)
        new_t.append(current_t)
        if current_t in t:
            for p in range(param_num):
                new_param[p].append(param_set[p][i])
            i += 1
        else:
            for p in range(param_num):
                new_param[p].append(np.mean([param_set[p][i],param_set[p][i+1]]))

    return new_t, new_param


def traj_process(location, start_end_time, folder_name, platoon_number):
    divided_location = available_in_all(location, start_end_time, platoon_number)
    part = 1
    traj_dict=[]
    for location in divided_location:
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

        distance = [[],[],[]]

        for tt in range(len(traj_with_coordinate[0])):
            veh_loc = []
            v_num=0
            for veh in location:
                veh_loc.append((traj_with_coordinate[4+v_num*2][tt], traj_with_coordinate[5+v_num*2][tt]))
                v_num += 1
            if len(distance[0]) == 0:
                distance[0].append(0)
            else:
                distance[0].append(traj_with_coordinate[1][tt-1] * 0.44704 / data_frequency + distance[0][-1])

            # distance[1].append(distance[0][-1]-cal_distance(veh_loc[0],veh_loc[1]))
            # distance[2].append(distance[1][-1]-cal_distance(veh_loc[1],veh_loc[2]))
            distance[1].append(distance[0][-1]-geod.distance(veh_loc[0],veh_loc[1]).m)
            distance[2].append(distance[1][-1]-geod.distance(veh_loc[1],veh_loc[2]).m)

        traj = traj + distance + heights
        traj = [item[:min([len(item1) for item1 in traj])] for item in traj]
        traj_dict.append(traj)
        save_traj(traj, folder_name, str(part)+'_'+str(platoon_number))
        part += 1
    return traj_dict

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


def speed_visulization(traj_dict, folder_name, MA_window = 2, extended_period = 50, overall=False):
    oscillation_set = read_oscillation_time(folder_name)
    save_continuous_CF_traj(traj_dict, oscillation_set, folder_name)

    traj_color = ['green', 'red', 'blue']

    stablization_level = []
    oscillation_info = []
    for traj in traj_dict:
        if MA_window > 0:
            for i in range(1, len(traj)):
                traj[i] = moving_average(traj[i], MA_window * data_frequency)
        if overall == True:
            Time_range = np.arange(ceil(min(traj[0]) / 60) * 60, max(traj[0]), 30)
            fig = plt.figure(figsize=(7, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_position([0.08, 0.2, 0.82, 0.7])
            for i in range(1, len(traj)):
                plt.plot(traj[0], traj[i], c=traj_color[i - 1], label='veh' + str(i))
            plt.legend()
            plt.xticks(Time_range)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('speed(mph)', fontsize=20)
            plt.grid(True)
            plt.show()
            continue
        divided_traj, selected_oscillation_set = traj_by_oscillation_manual(traj, oscillation_set)
        y_limit=[[20, 50], [30, 60], [50, 80]]
        traj_num = 0
        overall_traj = traj.copy()
        for traj in divided_traj:
            oscillations_LV = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
            extended_traj, non_used_value = traj_by_oscillation_manual(overall_traj, [(oscillations_LV[0][0]-extended_period,
                                                               oscillations_LV[0][0]+extended_period)])
            extended_traj = extended_traj[0]
            traj[0] = [tt - oscillations_LV[0][0] for tt in traj[0]]
            extended_traj[0] = [tt - oscillations_LV[0][0] for tt in extended_traj[0]]

            # try:
            #     os.stat(os.path.dirname(__file__) + folder_name + 'split_traj/')
            # except:
            #     os.mkdir(os.path.dirname(__file__) + folder_name + 'split_traj/')
            # save_traj(extended_traj, folder_name + 'split_traj/', 'oscillation_' + str(selected_oscillation_set[traj_num][-1]))
            # traj_num+=1
            # continue

            oscillations_LV = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
            oscillations_FV = oscillation_statistics(traj[0], traj[2], data_frequency, fluent=True)
            if len(traj) > 3:
                try:
                    oscillations_TV = oscillation_statistics(traj[0], traj[3], data_frequency, fluent=True)
                except:
                    oscillations_TV = [[]]
            if oscillations_LV[0][3] > 55:
                used_y_limit = y_limit[2]
                speed_group = 'high'
            elif oscillations_LV[0][3] > 40:
                used_y_limit = y_limit[1]
                speed_group = 'mid'
            else:
                used_y_limit = y_limit[0]
                speed_group = 'low'
            oscillation_info.append((oscillations_LV[0],oscillations_FV[0],
                                     oscillations_TV[0],selected_oscillation_set[traj_num][2:]+[speed_group]))
            # traj_num+=1
            # continue
            if extended_period is not None:
                width = 12
            else:
                width = 7
            stablization_before = \
                extended_traj[1][find_nearest_index(extended_traj[0],oscillations_LV[0][2]-30):
                                 find_nearest_index(extended_traj[0],oscillations_LV[0][2])]
            stablization_after = \
                extended_traj[1][find_nearest_index(extended_traj[0],oscillations_LV[0][4]):
                                 find_nearest_index(extended_traj[0],oscillations_LV[0][4]+30)]
            stablization_level.append([np.std(stablization_before),selected_oscillation_set[traj_num][2:]+[speed_group]])
            stablization_level.append([np.std(stablization_after),selected_oscillation_set[traj_num][2:]+[speed_group]])
            # print(stablization_level[-2])
            # print(stablization_level[-1])
            # traj_num+=1
            # continue
            fig = plt.figure(figsize=(width, 15), dpi=100)
            first_fig = 'oblique_traj'
            bx = fig.add_subplot(311)
            bx.set_position([0.08, 0.65, 0.87, 0.2])
            plt.xlim(extended_traj[0][0], extended_traj[0][-1])
            plt.grid(True)
            plt.title('OSCILLATION: ' + str(selected_oscillation_set[traj_num][-1]) +
                      ', HEADWAY: ' + selected_oscillation_set[traj_num][2] +
                      ', ENGINE MODE: ' + selected_oscillation_set[traj_num][3] +
                      '\nMAGNITUDE: ' + selected_oscillation_set[traj_num][4] +
                      ', BRAKE PATTERN: ' + selected_oscillation_set[traj_num][5] +
                      ', CRUISE: ' + selected_oscillation_set[traj_num][6].replace('\n',''))

            if first_fig == 'spacing':
                spacing_FV = [extended_traj[4][j] - extended_traj[5][j] for j in range(len(extended_traj[5]))]
                spacing_TV = [extended_traj[5][j] - extended_traj[6][j] for j in range(len(extended_traj[5]))]
                mean_spacing_FV = np.mean(spacing_FV)
                mean_spacing_TV = np.mean(spacing_TV)
                plt.plot(extended_traj[0], [a - mean_spacing_FV for a in spacing_FV], 'r')
                plt.plot(extended_traj[0], [a - mean_spacing_TV for a in spacing_TV], 'b')
                plt.ylabel('normalized spacing (m)', fontsize=20)
                plt.ylim([-20, 20])

            if first_fig == 'oblique_traj':
                max_position = 0
                min_position = 0
                slope = (extended_traj[4][-1] - extended_traj[4][0]) / (len(extended_traj[4]) - 1)
                for i in range(1, 4):
                    oblique_traj = [extended_traj[i+3][j] - extended_traj[6][0] - slope * j
                                    for j in range(len(extended_traj[i+3]))]
                    color_indicator = np.array(extended_traj[i])
                    points = np.array([np.array(extended_traj[0]), np.array(oblique_traj)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(used_y_limit[0] + 5, used_y_limit[1] - 15)
                    lc = LineCollection(segments, cmap='jet_r', norm=norm)
                    lc.set_array(color_indicator)
                    lc.set_linewidth(4)
                    bx.add_collection(lc)
                    max_position = max(max(oblique_traj), max_position)
                    min_position = min(min(oblique_traj), min_position)
                plt.text(extended_traj[0][0], min_position - 5, '$v_0$='+str(round(slope*10/0.44704,2))+'mph')
                plt.ylabel('oblique location(m)', fontsize=20)
                plt.ylim([min_position - 10, max_position + 10])
                cmap_jet = pcm.get_cmap('jet_r')
                sm = plt.cm.ScalarMappable(cmap=cmap_jet,
                                           norm=plt.Normalize(vmin=used_y_limit[0] + 5, vmax=used_y_limit[1] - 15))
                cbar = plt.colorbar(sm, orientation='horizontal', cax=plt.axes([0.2, 0.595, 0.65, 0.025]))
                cbar.set_label('speed (mph)', fontsize=16)
                cbar.set_ticks([used_y_limit[0] + 5, used_y_limit[0] + 10, used_y_limit[0] + 15])

            ax = fig.add_subplot(312)
            ax.set_position([0.08, 0.35, 0.87, 0.2])
            for i in range(1, 4):
                ax.plot(extended_traj[0], extended_traj[i], c=traj_color[i - 1], label='veh' + str(i))
            for o in oscillations_LV:
                ax.scatter(o[2], o[3], color='g', s=60)
                ax.scatter(o[6], o[7], color='g', s=60)
                ax.scatter(o[8], o[9], color='g', s=60)
                ax.scatter(o[4], o[5], color='g', s=60)
                ax.scatter(o[0], o[1], color='k', marker='*', s=60)
            for o in oscillations_FV:
                ax.scatter(o[2], o[3], color='r', s=36)
                ax.scatter(o[6], o[7], color='r', s=36)
                ax.scatter(o[8], o[9], color='r', s=36)
                ax.scatter(o[4], o[5], color='r', s=36)
            for o in oscillations_TV:
                if len(o) == 0:
                    continue
                if o[2] > oscillations_FV[0][4]:
                    continue
                ax.scatter(o[2], o[3], color='b', s=36)
                ax.scatter(o[6], o[7], color='b', s=36)
                ax.scatter(o[8], o[9], color='b', s=36)
                ax.scatter(o[4], o[5], color='b', s=36)
            plt.xlim(extended_traj[0][0], extended_traj[0][-1])
            plt.ylim(used_y_limit)
            plt.ylabel('speed(mph)', fontsize=20)
            plt.grid(True)
            plt.legend(loc=1)
            # plt.text(extended_traj[0][0], used_y_limit[0]+8,
            #          'pre stabilization STD:'+str(round(np.std(stablization_before),2))+'$mph$')
            # plt.text(extended_traj[0][0], used_y_limit[0]+5,
            #          'post stabilization STD:'+str(round(np.std(stablization_after),2))+'$mph$')

            cx = fig.add_subplot(313)
            cx.set_position([0.08, 0.1, 0.87, 0.2])
            for i in range(1, 4):
                cx.plot(extended_traj[0], extended_traj[i+6], c=traj_color[i - 1], label='veh' + str(i))
            plt.legend(loc=1)
            plt.xlim(extended_traj[0][0], extended_traj[0][-1])
            plt.ylabel('elevation (m)', fontsize=20)
            plt.xlabel('time (s)', fontsize=20)
            plt.grid(True)
            plt.ylim([np.mean(extended_traj[7])-25, np.mean(extended_traj[7])+25])
            try:
                os.stat(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            except:
                os.mkdir(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            plt.savefig(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/split_' +
                        str(selected_oscillation_set[traj_num][-1]) + '.png')
            plt.close()
            traj_num += 1
    # print('average stabilization STD:',np.average([a[0] for a in stablization_level]))
    file = open(os.path.dirname(__file__) + folder_name + 'oscillation_info','wb')
    pickle.dump(oscillation_info, file)
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
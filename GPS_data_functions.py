# import folium
import os
import time
import numpy as np
# import branca.colormap as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import rc
from math import ceil, floor
from Analyze_functions_multiple_veh import overlap_period
from base_functions import find_nearest_index, moving_average
from oscillation_functions import oscillation_statistics, traj_by_oscillation, save_oscillations

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

def read_data_from_seperated_csv(file_name,Lat_column,Lon_column,speed_column,time_column):
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
                if len(tmp) < max([Lat_column, Lon_column, speed_column]):
                    break
                location[veh].append([float(tmp[Lat_column]), float(tmp[Lon_column]),
                                      float(tmp[speed_column])*2.23694, float(tmp[time_column])])
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

def fix_missing_GPS_frame(t,v):
    new_v = []
    new_t = []
    t_range = range(int(t[0]*data_frequency), int(t[-1]*data_frequency) + 1)
    i = 0
    for tt in t_range:
        current_t = round(tt/data_frequency,1)
        new_t.append(current_t)
        if current_t in t:
            new_v.append(v[i])
            i += 1
        else:
            new_v.append(np.mean([v[i],v[i+1]]))
    return new_t,new_v


def traj_process(location, start_end_time, folder_name, platoon_number, MA_window):
    divided_location = available_in_all(location, start_end_time, platoon_number)
    split = 1
    part = 1
    traj_dict=[]
    for location in divided_location:
        traj = []
        for veh in location:
            t = [location[veh][i][3] for i in range(len(location[veh]))]
            v = [location[veh][i][2] for i in range(len(location[veh]))]
            t, v = fix_missing_GPS_frame(t, v)
            if MA_window > 0:
                v = moving_average(v, MA_window * data_frequency)
            if len(traj) == 0:
                traj.append(t)
                traj.append(v)
            else:
                traj.append(v)
        traj = [item[:min([len(item1) for item1 in traj])] for item in traj]
        traj_dict.append(traj)
        save_traj(traj, folder_name, str(part)+'_'+str(platoon_number))
        part += 1
    return traj_dict

def speed_visulization(traj_dict, folder_name, overall=False):
    traj_color = ['green', 'red', 'blue']
    split = 1
    for traj in traj_dict:
        if overall == True:
            Time_range = np.arange(ceil(min(traj[0]) / 60) * 60, max(traj[0]), 30)
            # fig = go.Figure()
            # for i in range(1, len(traj)):
            #     fig.add_trace(go.Scatter(x=traj[0], y=traj[i], name='veh' + str(i),
            #                              line=dict(color=traj_color[i - 1], width=4)))
            # fig.update_layout(xaxis = dict(tickmode='array',tickvals=Time_range,
            #                                ticktext=[str(floor(divmod(tr,24*60*60)[1]/60/60)-4)+':'+
            #                                          str(round(divmod(tr,60*60)[1]/60,1))  for tr in Time_range]))
            # fig.show(renderer="browser")
            # time.sleep(1)
            # input("Press Enter to continue...")
            fig = plt.figure(figsize=(7, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_position([0.15, 0.2, 0.82, 0.7])
            for i in range(1, len(traj)):
                plt.plot(traj[0], traj[i], c=traj_color[i - 1], label='veh' + str(i))
            plt.legend()
            plt.xticks(Time_range)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('speed(mph)', fontsize=20)
            plt.grid(True)
            plt.show()
            continue

        extended_period = 80
        divided_traj, extended_divided_traj = traj_by_oscillation_manual(traj, folder_name, extended_period)
        y_limit=[[20, 50], [30, 60], [50, 80]]
        traj_num = 0
        for traj in divided_traj:
            extended_traj = extended_divided_traj[traj_num]
            traj_num += 1
            traj[0] = [tt - extended_traj[0][0] for tt in traj[0]]
            extended_traj[0] = [tt - extended_traj[0][0] for tt in extended_traj[0]]

            oscillations_LV = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
            oscillations_FV = oscillation_statistics(traj[0], traj[2], data_frequency, fluent=True)
            if len(traj) > 3:
                try:
                    oscillations_TV = oscillation_statistics(traj[0], traj[3], data_frequency, fluent=True)
                except:
                    oscillations_TV = []
            save_oscillations(oscillations_FV, oscillations_LV, '', '', split, os.getcwd() + folder_name)
            print('printing:', split)
            if extended_period is not None:
                width = 12
            else:
                width = 7
            fig = plt.figure(figsize=(width, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_position([0.15, 0.2, 0.82, 0.7])
            for i in range(1, len(extended_traj)):
                plt.plot(extended_traj[0], extended_traj[i], c=traj_color[i - 1], label='veh' + str(i))
            # for i in range(1, len(traj)):
            #     plt.plot(traj[0], traj[i], c = traj_color[i-1], label='veh' + str(i))
            for o in oscillations_LV:
                plt.scatter(o[6], o[7], color='g', s=60)
                plt.scatter(o[8], o[9], color='g', s=60)
                plt.scatter(o[2], o[3], color='g', s=60)
                plt.scatter(o[4], o[5], color='g', s=60)
                plt.scatter(o[0], o[1], color='k', marker='*', s=60)
            # for o in oscillations_FV:
            #     plt.scatter(o[6], o[7], color='r', s=60)
            #     plt.scatter(o[8], o[9], color='r', s=60)
            #     plt.scatter(o[2], o[3], color='r', s=60)
            #     plt.scatter(o[4], o[5], color='r', s=60)
            #     plt.scatter(o[0], o[1], color='k', marker='*', s=60)

            # if len(traj) > 3:
            #     for o in oscillations_TV:
            #         plt.scatter(o[6], o[7], color='b', s=60)
            #         plt.scatter(o[8], o[9], color='b', s=60)
            #         plt.scatter(o[2], o[3], color='b', s=60)
            #         plt.scatter(o[4], o[5], color='b', s=60)
            #         plt.scatter(o[0], o[1], color='k', marker='*', s=60)
            plt.legend()
            if oscillations_LV[0][3] > 55:
                used_y_limit = y_limit[2]
            elif oscillations_LV[0][3] > 40:
                used_y_limit = y_limit[1]
            else:
                used_y_limit = y_limit[0]
            plt.ylim(used_y_limit)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('speed(mph)', fontsize=20)
            plt.title('oscillation: '+str(split))
            try:
                os.stat(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            except:
                os.mkdir(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/')
            plt.savefig(os.path.dirname(__file__) + folder_name + 'figures_GPS_data/split_' + str(split) + '.png')
            plt.close()
            split += 1

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

def traj_by_oscillation_manual(traj, folder_name, extended_period=100):
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
            oscillation_set.append((int(tmp[0]), int(tmp[1])))
        except:
            print('missing one', tmp[8], tmp[9], tmp[10], tmp[11])
    fo.close()
    divided_traj=[]
    extended_divided_traj = []

    for oscillation in oscillation_set:
        if (oscillation[0] > traj[0][-1]) or (oscillation[1] < traj[0][0]):
            continue
        s=find_nearest_index(traj[0],max(traj[0][0],oscillation[0]))
        e=find_nearest_index(traj[0],min(traj[0][-1],oscillation[1]))
        divided_traj.append([i[s:e] for i in traj])
        if extended_period == None:
            extended_divided_traj.append([i[s:e] for i in traj])
        else:
            mid_point = np.mean([oscillation[1], oscillation[0]])
            s_ex = find_nearest_index(traj[0], max(traj[0][0], mid_point - extended_period / 2))
            e_ex = find_nearest_index(traj[0], min(traj[0][-1], mid_point + extended_period / 2))
            extended_divided_traj.append([i[s_ex:e_ex] for i in traj])
    return divided_traj, extended_divided_traj

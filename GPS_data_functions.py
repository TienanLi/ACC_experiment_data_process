import folium
import os
import numpy as np
import branca.colormap as cm
import matplotlib.pyplot as plt
from Analyze_functions_multiple_veh import overlap_period
from base_functions import find_nearest_index, moving_average
from oscillation_functions import oscillation_statistics, traj_by_oscillation

global data_frequency
data_frequency = 10

def read_data_from_summary_csv(folder_name):
    traj_dict=[]
    for csv_file in os.listdir(os.path.dirname(__file__) + folder_name):
        if '.csv' not in csv_file:
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
        location[veh] = []
        start_end_time[veh] = []
        for csv_file in os.listdir(os.path.dirname(__file__)+file_name+'/'+veh):
            if '.csv' not in csv_file:
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


def traj_process(location, start_end_time, folder_name):
    divided_location = available_in_all(location, start_end_time, 2)
    split = 1
    part = 1
    traj_dict=[]
    for location in divided_location:
        traj = []
        for veh in location:
            t = [location[veh][i][3] for i in range(len(location[veh]))]
            v = [location[veh][i][2] for i in range(len(location[veh]))]
            t, v = fix_missing_GPS_frame(t, v)
            v = moving_average(v, 1 * data_frequency)
            if len(traj) == 0:
                traj.append(t)
                traj.append(v)
            else:
                traj.append(v)
        traj = [item[:min([len(item1) for item1 in traj])] for item in traj]
        traj_dict.append(traj)
        save_traj(traj, folder_name, part)
        part += 1
    return traj_dict

def speed_visulization(traj_dict):
    split = 1
    part = 1
    for traj in traj_dict:
        part += 1
        oscillations = oscillation_statistics(traj[0], traj[1], data_frequency, fluent=True)
        divided_traj = traj_by_oscillation(traj, oscillations, extended_time=45)
        for traj in divided_traj:
            print('printing:', split)
            fig = plt.figure(figsize=(12, 8), dpi=100)
            try:
                os.stat('figures_GPS_data/')
            except:
                os.mkdir('figures_GPS_data/')
            for i in range(1, len(traj)):
                plt.plot(traj[0], traj[i], label='veh' + str(i))
            plt.legend()
            plt.savefig('figures_GPS_data/split_' + str(split) + '.png')
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


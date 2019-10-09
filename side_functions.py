import matplotlib.pyplot as plt
import os
import numpy as np
from math import ceil,floor

def draw_fig(x,x_label,y,y_label):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    plt.plot(x, y)
    plt.xlabel(x_label,fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.savefig(y_label+'.png')
    plt.close()

def hex_to_byte(hex_str, length):

    scale = 16  ## equals to hexadecimal
    num_of_bits = length
    return bin(int(hex_str, scale))[2:].zfill(num_of_bits)


def hex_to_int(hex_str,total_length,m_start,m_length,signed):
    int_value=int(hex_to_byte(hex_str,total_length)[m_start:m_start+m_length],2)
    if signed and hex_str[0]==1:
        return ~int_value
    return int_value

def read_data_from_csv(file_name,message_ID_location):

    information={}
    fo = open(os.path.dirname(__file__)+file_name, 'r')
    fo.readline()
    line_num=0
    while True:
        line_num+=1
        #for each line
        line = fo.readline()
        if not line:
            break
        #split the whole line by comma
        tmp = line.split(',')

        if message_ID_location==1:
            if len(tmp) < 4:
                break
            time=line_num
            BUS=tmp[0]
            message_ID=tmp[1]
            message=tmp[2]
            try:
                message_length = int(tmp[3].replace("\n", ""))
            except:
                message_length = 0
        elif message_ID_location==5:
            if len(tmp) < 8:
                break
            time=int(tmp[0])*60*60+int(tmp[1])*60+int(tmp[2])+int(tmp[3])/1e6
            BUS=tmp[4]
            message_ID=int(tmp[5].replace('L',''), 16)
            message=tmp[6]
            message_length=tmp[7]


        if message_ID in information.keys():
            information[message_ID].append((time,message,message_length,BUS))
        else:
            information[message_ID]=[]
            information[message_ID].append((time,message,message_length,BUS))

        if line_num>1e6:
            break

    fo.close()
    return information

def draw_traj(speed_time,speed,front_space_time,front_space,fig_name):
    original_location=[0]
    for i in range(len(speed)-1):
        forward = (speed_time[i + 1] - speed_time[i]) * speed[i] / 3.6
        original_location.append(original_location[-1]+forward) #in meter


    new_time_start=ceil(max(speed_time[0],front_space_time[0]))
    new_time_end=floor(min(speed_time[-1],front_space_time[-1]))
    expected_frequency=100
    new_time_range=np.arange(new_time_start,new_time_end,1/expected_frequency)

    t,d=convert_time_series_frequency(speed_time,original_location,new_time_range)
    t,g=convert_time_series_frequency(front_space_time,front_space,new_time_range)
    d_LV=[d[i]+g[i] for i in range(len(d))]


    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    plt.plot(t, d, color='r', label='FV')
    plt.plot(t, d_LV, color='g', label='LV')
    plt.xlabel('time', fontsize=24)
    plt.ylabel('location', fontsize=24)
    plt.legend()
    plt.savefig('traj_'+fig_name + '.png')
    plt.close()

def convert_time_series_frequency(time_series,y_data,new_time_series):
    new_y_data=[]

    new_s_i=0
    for i in range(len(time_series)-1):
        interplot_start=time_series[i]
        interplot_end=time_series[i+1]
        if interplot_start==interplot_end:
            continue
        y_start=y_data[i]
        y_end=y_data[i+1]
        slope=(y_end-y_start)/(interplot_end-interplot_start)
        while new_time_series[new_s_i]>=interplot_start and new_time_series[new_s_i]<=interplot_end:
            x=new_time_series[new_s_i]-interplot_start
            new_y_data.append(y_start+x*slope)
            new_s_i+=1
            if new_s_i>=len(new_time_series):
                break
        if new_s_i >= len(new_time_series):
            break

    return new_time_series,new_y_data
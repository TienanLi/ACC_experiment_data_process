import matplotlib.pyplot as plt
import os
import numpy as np
from math import ceil,floor
import pandas as pd

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
    t=np.arange(new_time_start,new_time_end,1/expected_frequency)
    v=convert_time_series_frequency(speed_time,speed,t)
    d=convert_time_series_frequency(speed_time,original_location,t)
    g=convert_time_series_frequency(front_space_time,front_space,t)
    d_LV=[d[i]+g[i] for i in range(len(d))]
    v_LV=[(d_LV[i+1]-d_LV[i])/0.01*3.6 for i in range(len(d_LV)-1)]
    v_LV.append(v_LV[-1])
    v_LV=moving_average(v_LV,200)
    t_ita,ita=cal_ita(t,d_LV,t,d,sim_freq=0.01,w=5,k=0.1333)


    fig = plt.figure(figsize=(8, 12), dpi=300)

    ax = fig.add_subplot(311)
    plt.plot(t, d, color='r', label='FV')
    plt.plot(t, d_LV, color='g', label='LV')
    plt.ylabel('location(m)', fontsize=24)
    plt.legend()
    plt.xlim([t[0]+3,t[-1]])

    ax = fig.add_subplot(312)
    plt.plot(t_ita, ita, color='b')
    plt.ylabel(r'$\eta$', fontsize=24)
    plt.xlim([t[0]+3, t[-1]])
    plt.ylim([0.5,2])

    ax = fig.add_subplot(313)
    plt.plot(t, v, color='r', label='FV')
    plt.plot(t, v_LV, color='g', label='LV(derived from location)')
    plt.xlabel('time (s)', fontsize=24)
    plt.ylabel('speed(kph)', fontsize=24)
    plt.legend()
    plt.xlim([t[0] + 3, t[-1]])
    plt.ylim([0,80])

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

    return new_y_data

def cal_ita(t,d,t_f,d_f,sim_freq,w,k):
    ita=[]
    t_ita=[]
    for i in range(len(t_f)):
        try:
            r=range(int(max(0, (w_function(t_f[i]-t[0], w, 1, k)-3)  / sim_freq)),
                      int(min(len(t_f), (w_function(t_f[i]-t[0], w, 1, k)+3) / sim_freq)))
            y_list=[abs(-w*(t[j]-t_f[i])+d_f[i]-d[j]) for j in r]
            tau_i=t_f[i]-(y_list.index(min(y_list))*sim_freq+t[r[0]])
            eta=tau_i / ((1 / k / w))
            # eta = tau_i
            if eta>5 or eta<0:
                eta=np.nan
            ita.append(eta)
            t_ita.append(t_f[i])
        except:
            break
    return t_ita,ita

def w_function(x,w,ita,k):
    y=(x-(1 / w / k)*ita)
    return y

def moving_average(a, n) :
    return pd.Series(a).rolling(n, min_periods=9).mean().tolist()
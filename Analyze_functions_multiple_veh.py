import matplotlib.pyplot as plt
import os
import numpy as np
from base_functions import draw_fig,cal_ita,moving_average,divide_traj,find_nearest_index
from Analyze_functions import read_data_from_csv,fill_front_space_missing_signal,analyze_CANBUS

global expected_frequency
expected_frequency = 100


def analyze_and_draw_2(messeage_dict,run,front_name,follow_name):
    traj_info_l=[]
    traj_info_f=[]
    for messeages in messeage_dict[front_name]:
        info_list = analyze_CANBUS(messeages, 'prius')
        traj_info_l.append(info_list)
    for messeages in messeage_dict[follow_name]:
        info_list = analyze_CANBUS(messeages, 'prius')
        traj_info_f.append(info_list)
    traj_info=find_overlapping(traj_info_l,traj_info_f)
    part=1
    for traj in traj_info:
        divided_traj=divide_traj(traj,period_length=1e8)
        split=1
        try:
            os.stat('figures_2/')
        except:
            os.mkdir('figures_2/')
        for (time_series, f_speed, f_front_space, f_relative_speed, l_speed, l_front_space, l_relative_speed) in divided_traj:
            draw_traj_2(time_series,f_speed,f_front_space,f_relative_speed,l_speed,l_front_space,l_relative_speed,
                        'figures_2/'+str(run)+'_part'+str(part)+'_split'+str(split))
            split+=1
        part += 1

def read_two_vehicle_data(folder,run,front_name,follow_name):
    information={}
    information[front_name]=[]
    information[follow_name]=[]
    for element in os.listdir(os.path.dirname(__file__)+folder):
        if str(run) in element:
            for vehicle in [front_name,follow_name]:
                if vehicle in element:
                    information[vehicle].append(read_data_from_csv(folder+element,5))
    return information

def draw_traj_2(t,f_speed,f_front_space,f_relative_speed,l_speed,l_front_space,l_relative_speed,fig_name):
    d=[0]
    for i in range(len(f_speed)-1):
        forward = (t[i + 1] - t[i]) * f_speed[i] / 3.6
        d.append(d[-1]+forward) #in meter

    draw_fig(t,'',f_front_space,'original space (m)')
    f_front_space=fill_front_space_missing_signal(f_front_space,high_threshold=100)
    # l_front_space=fill_front_space_missing_signal(l_front_space,high_threshold=200)
    draw_fig(t,'',f_front_space,'revised space (m)')

    v_LV_back_measured=[f_speed[i]+f_relative_speed[i] for i in range(len(f_speed))]
    diff=np.mean(v_LV_back_measured)-np.mean(l_speed)
    l_speed_revised=[s+diff for s in l_speed]

    d_LV=[d[i]+f_front_space[i] for i in range(len(d))]

    d_LV_derived=[d[0]+f_front_space[0]]
    for i in range(len(d)-1):
        d_LV_derived.append(d_LV_derived[i]+(f_speed[i]+f_relative_speed[i])/3.6*0.01)
    diff=np.mean(d_LV)-np.mean(d_LV_derived)
    d_LV_derived=[d+diff for d in d_LV_derived]

    d_LV_front_measured=[d[0]+f_front_space[0]]
    for i in range(len(d)-1):
        d_LV_front_measured.append(d_LV_front_measured[i]+l_speed_revised[i]/3.6*0.01)
    diff=np.mean(d_LV)-np.mean(d_LV_front_measured)
    d_LV_front_measured=[d+diff for d in d_LV_front_measured]

    v_LV_derived = [(d_LV[i + 1] - d_LV[i-1]) / 0.01 * 3.6/2 for i in range(1,len(d_LV) - 1)]
    v_LV_derived=[v_LV_derived[0]]+v_LV_derived+[v_LV_derived[-1]]
    v_LV_derived = moving_average(v_LV_derived, 200)

    t_ita,ita=cal_ita(t,d_LV,t,d,sim_freq=0.01,w=5,k=0.1333)
    t_ita_derived,ita_derived=cal_ita(t,d_LV_derived,t,d,sim_freq=0.01,w=5,k=0.1333)
    t_ita_front_measured,ita_front_measured=cal_ita(t,d_LV_front_measured,t,d,sim_freq=0.01,w=5,k=0.1333)


    fig = plt.figure(figsize=(8, 12), dpi=300)
    ax = fig.add_subplot(311)
    plt.plot(t, d, color='r', label='FV')
    plt.plot(t, d_LV, color='g', label='LV (direct measured from spacing)')
    plt.plot(t, d_LV_derived, color='k', label='LV (integrated from relative speed)')
    plt.plot(t, d_LV_front_measured, color='c', label='LV (integrated from shifted lead self speed)')
    plt.ylabel('location(m)', fontsize=24)
    plt.legend()
    plt.xlim([t[0]+3,t[-1]])

    ax = fig.add_subplot(312)
    plt.plot(t_ita, ita, color='g',label='direct measured from spacing')
    plt.plot(t_ita_derived, ita_derived, color='k', label='integrated from relative speed')
    plt.plot(t_ita_front_measured, ita_front_measured, color='c', label='integrated from shifted lead self speed')
    plt.ylabel(r'$\eta$', fontsize=24)
    plt.xlim([t[0]+3, t[-1]])
    plt.ylim([0.5,2])
    plt.legend()

    ax = fig.add_subplot(313)
    plt.plot(t, f_speed, color='r', label='FV')
    plt.plot(t, v_LV_derived, color='g', label='LV (derived from distance)')
    plt.plot(t, v_LV_back_measured, color='k', label='LV (measured from FV radar)')
    # plt.plot(t, l_speed, color='b', label='LV (direct measured from LV CANBUS)')
    plt.plot(t, l_speed_revised, color='c', label='LV (LV CANBUS shift)')

    plt.xlabel('time (s)', fontsize=24)
    plt.ylabel('speed(kph)', fontsize=24)
    plt.legend()
    plt.xlim([t[0] + 3, t[-1]])
    plt.ylim([max(0,np.mean(f_speed)-40),np.mean(f_speed)+40])
    plt.savefig(fig_name + '.png')
    plt.close()


def find_overlapping(traj_lead,traj_follow):
    time_period_lead=[]
    time_period_follow=[]
    traj_lead_connect=[[] for i in range(len(traj_lead[0]))]
    traj_follow_connect=[[] for i in range(len(traj_follow[0]))]
    for traj in traj_lead:
        time_period_lead.append((max(traj[0][0],traj[2][0]),min(traj[0][-1],traj[2][-1])))
        traj_lead_connect=[traj_lead_connect[i]+traj[i] for i in range(len(traj_lead_connect))]
    for traj in traj_follow:
        time_period_follow.append((max(traj[0][0],traj[2][0]),min(traj[0][-1],traj[2][-1])))
        traj_follow_connect=[traj_follow_connect[i]+traj[i] for i in range(len(traj_follow_connect))]
    shared_period=overlap_period(time_period_lead,time_period_follow)
    traj_info=[]
    for period in shared_period:
        time_series=np.arange(period[0],period[1],1/expected_frequency)
        index_in_l_speed=find_nearest_index(traj_lead_connect[0],time_series[0])
        index_in_l_radar=find_nearest_index(traj_lead_connect[2],time_series[0])
        index_in_f_speed=find_nearest_index(traj_follow_connect[0],time_series[0])
        index_in_f_radar=find_nearest_index(traj_follow_connect[2], time_series[0])
        f_speed=traj_follow_connect[1][index_in_f_speed:index_in_f_speed+len(time_series)]
        f_front_space=traj_follow_connect[3][index_in_f_radar:index_in_f_radar+len(time_series)]
        f_relative_speed=traj_follow_connect[4][index_in_f_radar:index_in_f_radar+len(time_series)]
        l_speed=traj_lead_connect[1][index_in_l_speed:index_in_l_speed+len(time_series)]
        l_front_space=traj_lead_connect[3][index_in_l_radar:index_in_l_radar+len(time_series)]
        l_relative_speed=traj_lead_connect[4][index_in_l_radar:index_in_l_radar+len(time_series)]
        traj_info.append((time_series,f_speed,f_front_space,f_relative_speed,l_speed,l_front_space,l_relative_speed))
    return traj_info

def overlap_period(period_set_1,period_set_2):
    overlapping_period=[]
    i1=0
    i2=0
    while True:
        start=max(period_set_1[i1][0],period_set_2[i2][0])
        end=min(period_set_1[i1][1],period_set_2[i2][1])
        if end-start>30:
            overlapping_period.append((start,end))
        if  end==period_set_1[i1][1]:
            i1+=1
        else:
            i2+=1
        if i2>=len(period_set_2) or i1>=len(period_set_1):
            return overlapping_period
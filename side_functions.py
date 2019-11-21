import matplotlib.pyplot as plt
import os
import numpy as np
from math import ceil,floor
import pandas as pd
from civic import analyze_ENGINE_DATA,analyze_KINEMATICS
from prius import analyze_SPEED,analyze_PCM_CRUISE,analyze_LEAD_INFO
from base_functions import draw_fig,convert_time_series_frequency

global expected_frequency
expected_frequency = 100

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

def read_data_from_csv(file_name,message_ID_location):
    information={}
    information[466] = []
    information[742] = []
    information[180] = []
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
            try:
                time=int(tmp[0])*60*60+int(tmp[1])*60+int(tmp[2])+int(tmp[3])/1e6
                BUS=tmp[4]
                message_ID=int(tmp[5].replace('L',''), 16)
                message=tmp[6]
                message_length=tmp[7]
            except:
                break
        # if message_ID in information.keys():
        #     information[message_ID].append((time,message,message_length,BUS))
        # else:
        #     information[message_ID]=[]
        #     information[message_ID].append((time,message,message_length,BUS))
        if message_ID in [466,742,180]:
            information[message_ID].append((time,message,message_length,BUS))
    fo.close()
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
    plt.ylim([0.5,3])
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

def traj_derivation(traj):
    (speed_time, speed, front_space_time, front_space, relative_speed)=traj
    original_location = [0]
    for i in range(len(speed) - 1):
        forward = (speed_time[i + 1] - speed_time[i]) * speed[i] / 3.6
        original_location.append(original_location[-1] + forward)  # in meter
    t = speed_time
    v = speed
    d = original_location
    draw_fig(t, '', front_space, 'space(m)')
    front_space = fill_front_space_missing_signal(front_space, high_threshold=100)
    space = front_space
    # r_v=relative_speed
    draw_fig(t, '', space, 'revised space (m)')
    d_LV = [d[i] + space[i] for i in range(len(d))]
    # d_LV=moving_average(d_LV,100)
    # d_LV_derived=[d[0]+space[0]]
    # for i in range(len(d)-1):
    #     d_LV_derived.append(d_LV_derived[i]+(v[i]+r_v[i])/3.6*0.01)
    # diff=np.mean(d_LV)-np.mean(d_LV_derived)
    # d_LV_derived=[d+diff for d in d_LV_derived]
    t_ita, ita = cal_ita(t, d_LV, t, d, sim_freq=1/expected_frequency, w=5, k=0.1333)
    # t_ita_derived,ita_derived=cal_ita(t,d_LV_derived,t,d,sim_freq=0.01,w=5,k=0.1333)
    # v_LV_measured=[v[i]+r_v[i] for i in range(len(v))]
    v_LV_derived = [(d_LV[i + 1] - d_LV[i - 1]) * expected_frequency * 3.6 / 2 for i in range(1, len(d_LV) - 1)]
    v_LV_derived = [v_LV_derived[0]] + v_LV_derived + [v_LV_derived[-1]]
    v_LV_derived = moving_average(v_LV_derived, 2*expected_frequency)
    return t,v,d,v_LV_derived,d_LV,t_ita,ita

def draw_traj(t,v,d,v_LV_derived,d_LV,t_ita,ita,oscillations,fig_name):
    fig = plt.figure(figsize=(8, 12), dpi=300)
    ax = fig.add_subplot(311)
    plt.plot(t, d, color='r', label='FV')
    plt.plot(t, d_LV, color='g', label='LV (direct measured from radar)')
    # plt.plot(t, d_LV_derived, color='k', label='LV (integrated from relative speed)')
    plt.ylabel('location(m)', fontsize=24)
    plt.legend()
    plt.xlim([t[0]+3,t[-1]])
    ax = fig.add_subplot(312)
    plt.plot(t_ita, ita, color='g',label='direct measured from radar')
    # plt.plot(t_ita_derived, ita_derived, color='k', label='integrated from relative speed')
    plt.ylabel(r'$\eta$', fontsize=24)
    plt.xlim([t[0]+3, t[-1]])
    plt.ylim([0.5,3])
    plt.legend()
    ax = fig.add_subplot(313)
    plt.plot(t, v, color='r', label='FV')
    plt.plot(t, v_LV_derived, color='g', label='LV (derived from spacing)')
    # plt.plot(t, v_LV_measured, color='k', label='LV (direct measured from radar)')
    for o in oscillations:
        plt.scatter(o[0],o[1],color='r*')
        plt.scatter(o[2],o[3],color='b')
        plt.scatter(o[4],o[5],color='b')
        plt.scatter(o[6],o[7],color='b')
        plt.scatter(o[8],o[9],color='b')
        plt.text(o[2],o[3],str(o[10])+'s\nd=-'+str(o[11])+'$m/s^2$')
        plt.text(o[4],o[5],str(o[12])+'s\na='+str(o[13])+'$m/s^2$')

    plt.xlabel('time (s)', fontsize=24)
    plt.ylabel('speed(kph)', fontsize=24)
    plt.legend()
    plt.xlim([t[0] + 3, t[-1]])
    plt.ylim([max(0,np.mean(v)-40),np.mean(v)+40])
    plt.savefig(fig_name + '.png')
    plt.close()


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
    return pd.Series(a).rolling(n, min_periods=int(n/5)).mean().tolist()

def fill_front_space_missing_signal(serie,high_threshold):
    missing_index=[]
    unnormal_threshold=100/expected_frequency
    unnormal=False
    unnormal_down=False

    # if serie[0]>=high_threshold:
    #     missing_index.append(0)
    for i in range(1,len(serie)):
        if  (not unnormal_down) and (serie[i] - serie[i - 1]) >= unnormal_threshold:
            unnormal=True
        if  (not unnormal) and(serie[i] - serie[i - 1]) <= - unnormal_threshold:
            unnormal_down=True
        if unnormal and (serie[i] - serie[i - 1]) <= -unnormal_threshold:
            unnormal=False
        if unnormal_down and (serie[i] - serie[i - 1]) >= unnormal_threshold:
            unnormal_down=False
        if unnormal or unnormal_down:
            missing_index.append(i)
        else:
            if len(missing_index) == 0:
                interplot_start = serie[i]
            else:
                interplot_end=serie[i]
                x_start=missing_index[0] - 1
                try:
                    slope = (interplot_end - interplot_start) / (i - x_start)
                    if abs(slope)>(2.5/expected_frequency):
                        missing_index.append(i)
                        if (serie[i] - serie[i - 1]) >= unnormal_threshold:
                            unnormal_down = True
                        if (serie[i] - serie[i - 1]) <= -unnormal_threshold:
                            unnormal = True
                    else:
                        for m_i in missing_index:
                            serie[m_i] = interplot_start + (m_i - x_start) * slope
                        missing_index = []
                except:
                    for m_i in missing_index:
                        serie[m_i] = interplot_end
                    missing_index=[]

    return serie

def get_ID_loc_and_model(run):
    messeage_ID_location=5
    model='prius'
    if run==5|7:
        model='carolla'
    if run<=3:
        model='civic'
        messeage_ID_location=1
    print('run:',run)
    return messeage_ID_location,model

def save_traj_info(t, v, d, v_LV_derived, d_LV,run,set,part,period=None):
    if period==None:
        t=[tt-t[0] for tt in t]
        flink = open('data/traj_output/run_%s_set_%s_part_%s.csv'%(run,set,part),'w')
        flink.write('time stamp(sec),follower location(m),follower speed(km/h),leader location(m),leader speed(km/h)\n')
        for i in range(len(t)):
            flink.write('%s,%s,%s,%s,%s\n' % (round(t[i],2), round(d[i],3), round(v[i],3), round(d_LV[i],3), round(v_LV_derived[i],3)))
        flink.close()
    else:
        sub=1
        for p in period:
            s_i=find_nearest_index(t,p[0])
            e_i=find_nearest_index(t,p[1])
            t_print=t[s_i:e_i]
            d_print=d[s_i:e_i]
            v_print=v[s_i:e_i]
            dlv_print=d_LV[s_i:e_i]
            vlv_print=v_LV_derived[s_i:e_i]
            t_print=[tt-t_print[0] for tt in t_print]
            dlv_print=[dd-d_print[0] for dd in dlv_print]
            d_print=[dd-d_print[0] for dd in d_print]
            flink = open('data/traj_output/run_%s_set_%s_part_%s_osc_%s.csv'%(run,set,part,sub),'w')
            flink.write('time stamp(sec),follower location(m),follower speed(km/h),leader location(m),leader speed(km/h)\n')
            for i in range(len(t_print)):
                flink.write('%s,%s,%s,%s,%s\n' % (round(t_print[i],2), round(d_print[i],3), round(v_print[i],3), round(dlv_print[i],3), round(vlv_print[i],3)))
            flink.close()
            sub+=1

def oscillation_statistics(t,v,d,v_LV,d_LV):
    v_threshold=np.percentile(v,15)
    oscillation_pair=[]
    oscillation=False

    for i in range(1,len(v)):
        if v[i]<v_threshold and v[i-1]>=v_threshold:
            start=i
            oscillation=True
        if oscillation and (v[i]>=v_threshold and v[i-1]<v_threshold):
            end=i
            oscillation=False
            if (end - start) > (1 * expected_frequency):
                oscillation_pair.append((start,end))

    oscillation_candidates=[v[p[0]:p[1]] for p in oscillation_pair]
    minimum_speed=[min(c) for c in oscillation_candidates]
    minimum_point=[c.index(min(c)) for c in oscillation_candidates]
    minimum_point_time_index=[oscillation_pair[i][0]+minimum_point[i] for i in range(len(oscillation_candidates))]
    minimum_point_time=[t[m] for m in minimum_point_time_index]

    oscillations=[[minimum_point_time[i],minimum_speed[i]] for i in range(len(oscillation_candidates))]
    #0: minimum speed time stamp
    #1: minimum speed

    o=0
    for t_p in minimum_point_time_index:
        previous_period=v[max(0,t_p-20*expected_frequency):t_p]
        following_period=v[t_p:min(len(t),t_p+150*expected_frequency)]
        previous_d=[previous_period[i]-previous_period[i-1] for i in range(1,len(previous_period))]
        previous_d=moving_average(previous_d,500)
        previous_d=[previous_d[0]]+previous_d
        following_a = [following_period[i] - following_period[i - 1] for i in range(1, len(following_period))]
        following_a = moving_average(following_a, 500)
        following_a=following_a+[following_a[-1]]

        minimum_d=0
        start_point=len(previous_period)-1
        idle_start=len(previous_period)-1
        for i in np.arange(len(previous_period)-2,1,-1):
            if (previous_period[i]-minimum_speed[o])<(v_threshold*0.05):
                idle_start=i
            if previous_period[i]>previous_period[start_point]:
                start_point=i
                if previous_d[i]<minimum_d:
                    minimum_d=previous_d[i]
            if (start_point-i)>2*expected_frequency or previous_d[i]>(minimum_d*0.05):
                break
        start_speed=previous_period[start_point]
        oscillations[o].append(t[t_p-len(previous_period)+start_point])
        oscillations[o].append(start_speed)
        #3 deceleration start
        #4 deceleration start speed

        maximum_a=0
        end_point=0
        idle_end=0
        for i in np.arange(1,len(following_period)):
            if (following_period[i]-minimum_speed[o])<(v_threshold*0.05):
                idle_end=i
            if following_period[i]>following_period[end_point]:
                end_point=i
                if following_a[i]>maximum_a:
                    maximum_a=following_a[i]
            if (i-end_point)>2*expected_frequency or following_a[i]<(maximum_a*0.05):
                break
        end_speed=following_period[end_point]
        oscillations[o].append(t[t_p+end_point])
        oscillations[o].append(end_speed)
        #4 acceleration end
        #5 acceleration end speed

        oscillations[o].append(t[t_p-len(previous_period)+idle_start])
        #6 deceleration end
        oscillations[o].append(t[t_p+idle_end])
        #7 acceleration begin
        oscillations[o].append(round(minimum_d/3.6*expected_frequency,2))
        #8 minimum deceleration rate
        oscillations[o].append(round(maximum_a/3.6*expected_frequency,2))
        #9 maximum acceleration rate

        o+=1

    for i in range(len(oscillations)):
        o=oscillations[i]
        deceleration_duration=o[0]-o[2]
        deceleration_rate=(o[3]-o[1])/3.6/deceleration_duration
        acceleration_duration=o[4]-o[2]
        acceleration_rate=(o[5]-o[1])/3.6/acceleration_duration
        oscillations[i]=oscillations[i]+[round(deceleration_duration,2),round(deceleration_rate,2),round(acceleration_duration,2),round(acceleration_rate,2)]
        #10 deceleration duration
        #11 avg decleration rate
        #12 accelereation duration
        #13 avg deceleration rate

    return oscillations

def analyze_and_draw(messeage_dict,model,run,set):
    [speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using]=analyze(messeage_dict,model)
    traj_info=(speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using)
    traj_info=ACC_in_use(speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using)
    part=1
    for traj in traj_info:
        t, v, d, v_LV_derived, d_LV, t_ita, ita=traj_derivation(traj)
        oscillations=oscillation_statistics(t,v,d,v_LV_derived,d_LV)

        # cut_points=[53520,53560,53555,53605]
        # period=[(cut_points[i*2],cut_points[i*2+1]) for i in range(int(len(cut_points)/2))]
        # period=None
        # save_traj_info(t, v, d, v_LV_derived, d_LV,run,set,part,period)

        divided_traj=divide_traj([t, v, d, v_LV_derived, d_LV, t_ita, ita],period_length=60)
        split=1
        for (t, v, d, v_LV_derived, d_LV, t_ita, ita) in divided_traj:
            try:
                os.stat('figures/' + str(run) +'/')
            except:
                os.mkdir('figures/' + str(run) +'/')
            draw_traj(t, v, d, v_LV_derived, d_LV, t_ita, ita,oscillations,
                      'figures/' + str(run) +'/'+str(run)+'_' + str(set) + '_part' + str(part)+'_split'+str(split))
            print('split:',split)
            split+=1

        part += 1

def analyze_and_draw_2(messeage_dict,run,front_name,follow_name):
    traj_info_l=[]
    traj_info_f=[]

    for messeages in messeage_dict[front_name]:
        info_list = analyze(messeages, 'prius')
        traj_info_l.append(info_list)
    for messeages in messeage_dict[follow_name]:
        info_list = analyze(messeages, 'prius')
        traj_info_f.append(info_list)
    traj_info=find_overlapping(traj_info_l,traj_info_f)

    part=1
    for traj in traj_info:
        divided_traj=divide_traj(traj,period_length=60)
        split=1
        for (time_series, f_speed, f_front_space, f_relative_speed, l_speed, l_front_space, l_relative_speed) in divided_traj:
            draw_traj_2(time_series,f_speed,f_front_space,f_relative_speed,l_speed,l_front_space,l_relative_speed,
                        'figures/'+str(run)+'_part'+str(part)+'_split'+str(split))
            split+=1
        part += 1


def divide_traj(traj,period_length):
    period_length=period_length*expected_frequency
    if len(traj[0])<=period_length:
        return [traj]
    divided_traj=[]
    i=0
    while True:
        s=i*int(period_length*0.75)
        i+=1
        e=s+period_length
        if e>(len(traj[0])-period_length):
            divided_traj.append([i[s:] for i in traj])
            break
        else:
            divided_traj.append([i[s:e] for i in traj])
    return divided_traj


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


def ACC_in_use(speed_time_series,speed,LEAD_INFO_time_series,front_space,relative_speed,ACC_using_ts,ACC_using):
    traj_pair=[]
    start = 0
    for i in range(len(ACC_using)-1):
        if ACC_using[i]==0 and ACC_using[i+1]==1:
            start=i
        elif ACC_using[i]==1 and ACC_using[i+1]==0:
            end=i
            if ACC_using_ts[end]-ACC_using_ts[start]>30:
                traj_pair.append((ACC_using_ts[start],ACC_using_ts[end]))
    end=len(ACC_using)-1
    if min(ACC_using_ts[end],speed_time_series[-1],LEAD_INFO_time_series[-1]) - max(ACC_using_ts[start],speed_time_series[0],LEAD_INFO_time_series[0]) > 25:
        traj_pair.append((max(ACC_using_ts[start],speed_time_series[0],LEAD_INFO_time_series[0]), min(ACC_using_ts[end],speed_time_series[-1],LEAD_INFO_time_series[-1])))

    traj_info=[]
    for p in traj_pair:
        ss=find_nearest_index(speed_time_series, p[0])
        se=find_nearest_index(speed_time_series, p[1])
        ls=find_nearest_index(LEAD_INFO_time_series, p[0])
        le=find_nearest_index(LEAD_INFO_time_series, p[1])
        traj_info.append((speed_time_series[ss:se],speed[ss:se],LEAD_INFO_time_series[ls:le],front_space[ls:le],relative_speed[ls:le]))
    return traj_info


def find_nearest_index(time_serires,point):
    series_a=[abs(ts-point) for ts in time_serires]
    return series_a.index(min(series_a))


def analyze(messeage_dict,model):
    if model=='civic':
        ENGINE_DATA=messeage_dict['0x158']
        analyze_ENGINE_DATA(ENGINE_DATA)
        KINEMATICS = messeage_dict['0x94'] # this is to get longi_accel
        analyze_KINEMATICS(KINEMATICS)
    else:
        SPEED=messeage_dict[180]
        speed_time_series,speed=analyze_SPEED(SPEED)
        LEAD_INFO1 = messeage_dict[466]
        ACC_using_ts, ACC_using = analyze_PCM_CRUISE(LEAD_INFO1)
        LEAD_INFO = messeage_dict[742]
        LEAD_INFO_time_series, front_space, relative_speed = analyze_LEAD_INFO(LEAD_INFO,742)
    return [speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using]
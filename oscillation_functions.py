import numpy as np
from base_functions import moving_average,find_nearest_index
import matplotlib.pyplot as plt
from base_functions import get_group_info,linear_regression
import numpy as np
from scipy import stats
from matplotlib import rc
import random
import string
font = {'family': 'DejaVu Sans',
        'size': 16}
rc('font', **font)

def oscillation_statistics(t,v,expected_frequency,fluent):
    # v_threshold=max(max(np.nanmax(v)*0.6,np.nanmean(v)*0.8),np.nanpercentile(v,15))
    # oscillation_pair=[]
    # oscillation=False
    # for i in range(1,len(v)):
    #     if v[i]<v_threshold and v[i-1]>=v_threshold:
    #         start=i
    #         oscillation=True
    #     if oscillation and (v[i]>=v_threshold and v[i-1]<v_threshold):
    #         end=i
    #         oscillation=False
    #         if (end - start) > (1 * expected_frequency):
    #             oscillation_pair.append((start,end))
    oscillation_pair = [(0,len(v))]
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
        idle_threshold = .5
        idle_threshold_rate = 1
        previous_period = v[max(0, t_p - 150 * expected_frequency):t_p]
        following_period = v[t_p:min(len(t), t_p + 150 * expected_frequency)]
        previous_d = [(previous_period[i] - previous_period[i - 1]) * expected_frequency for i in
                      range(1, len(previous_period))]
        if fluent:
            moving_period = 2 * expected_frequency
        else:
            moving_period = 4 * expected_frequency
        previous_d = moving_average(previous_d, moving_period)
        previous_d = [previous_d[0]] + previous_d
        following_a = [(following_period[i] - following_period[i - 1]) * expected_frequency for i in
                       range(1, len(following_period))]
        following_a = moving_average(following_a, moving_period)
        following_a = following_a + [following_a[-1]]

        idle_start, idle_start_speed, start_point, start_speed, minimum_d = \
            deceleration_parameters(minimum_speed, o, previous_period, previous_d,
                                    idle_threshold, idle_threshold_rate)

        oscillations[o].append(round(t[t_p-len(previous_period)+start_point],2))
        oscillations[o].append(round(start_speed,2))
        #2 deceleration start
        #3 deceleration start speed

        idle_end, idle_end_speed, end_point, end_speed, maximum_a = \
            acceleration_parameters(minimum_speed, o, following_period, following_a,
                                    idle_threshold, idle_threshold_rate, start_speed)

        oscillations[o].append(round(t[t_p+end_point],2))
        oscillations[o].append(round(end_speed,2))
        #4 acceleration end
        #5 acceleration end speed
        oscillations[o].append(round(t[t_p-len(previous_period)+idle_start],2))
        oscillations[o].append(round(idle_start_speed,2))
        #6 deceleration end
        #7 deceleration end speed
        oscillations[o].append(round(t[t_p+idle_end],2))
        oscillations[o].append(round(idle_end_speed,2))
        #8 acceleration begin
        #9 acceleration begin speed
        oscillations[o].append(-round(minimum_d * 0.44704 ,2)) #mph to m/s
        #10 minimum deceleration rate
        oscillations[o].append(round(maximum_a * 0.44704 ,2)) #mph to m/s
        #11 maximum acceleration rate

    for i in range(len(oscillations)):
        o=oscillations[i]
        deceleration_duration=o[6]-o[2]
        if deceleration_duration == 0:
            deceleration_rate = 1e8
        else:
            deceleration_rate = (o[3]-o[7]) / deceleration_duration * 0.44704 #mph to m/s
        acceleration_duration = o[4] - o[8]
        if acceleration_duration == 0:
            acceleration_rate = 1e8
        else:
            acceleration_rate=(o[5]-o[9])/acceleration_duration * 0.44704 #mph to m/s
        idle_duration=o[8]-o[6]
        oscillations[i]=oscillations[i]+[round(deceleration_duration,2),round(deceleration_rate,2),
                                         round(acceleration_duration,2),round(acceleration_rate,2),
                                         round(idle_duration,2)]
        # 12 deceleration duration
        # 13 avg decleration rate
        # 14 accelereation duration
        # 15 avg deceleration rate
        # 16 idle duration
    return oscillations

def deceleration_parameters(minimum_speed, o, previous_period, previous_d,
                            idle_threshold, idle_threshold_rate):
    minimum_d = 0
    start_point = len(previous_period) - 1
    idle_start = len(previous_period) - 1
    idle_start_speed = minimum_speed[o]
    for i in np.arange(len(previous_period) - 2, 1, -1):
        # recognize the deceleration start point
        if previous_period[i] > previous_period[start_point]:
            start_point = i
            if previous_d[i] < minimum_d:
                minimum_d = previous_d[i]
        if previous_d[i] > (minimum_d * 0.05) and minimum_d == min(previous_d):
            break
    start_speed = previous_period[start_point]

    idle_threshold_rate = abs(minimum_d / 3)
    for i in np.arange(len(previous_period) - 2, 1, -1):
        # recognize the idle start point
        if (abs(previous_d[i]) < idle_threshold_rate) or ((previous_period[i] - minimum_speed[o]) < idle_threshold):
            idle_start = i
            idle_start_speed = previous_period[i]
        if previous_d[i] == (minimum_d):
            break

    return idle_start, idle_start_speed, start_point, start_speed, minimum_d


def acceleration_parameters(minimum_speed, o, following_period, following_a,
                            idle_threshold, idle_threshold_rate, start_speed):
    maximum_a = 0
    end_point = 0
    idle_end = 0
    idle_end_speed = minimum_speed[o]
    for i in np.arange(1, len(following_period)):
        if following_period[i] >= following_period[end_point]:
            end_point = i
            if following_a[i] > maximum_a:
                maximum_a = following_a[i]
        if (maximum_a == max(following_a)) and (following_a[i] < (maximum_a * 0.05)):
            break
    end_speed = following_period[end_point]

    idle_threshold_rate = maximum_a / 3
    for i in np.arange(1, len(following_period)):
        if  (following_a[i] < idle_threshold_rate) or ((following_period[i] - minimum_speed[o]) < idle_threshold):
            idle_end = i
            idle_end_speed = following_period[i]
        if  following_a[i] > (maximum_a * .8):
            break

    return idle_end, idle_end_speed, end_point, end_speed, maximum_a

def save_oscillations(oscillation_FV,oscillation_LV,run,set,part, folder_name = ''):
    rsp_set=[]
    try:
        fo = open(folder_name+'oscillation_info.csv', 'r')
        while True:
            line = fo.readline()
            if not line:
                break
            tmp = line.split(',')
            rsp=tmp[0]+','+tmp[1]+','+tmp[2]+','
            rsp_set.append(rsp)
        fo.close()
        mode='a'
    except:
        mode='w'
    flink = open(folder_name+'oscillation_info.csv', mode)
    if 'run,set,part,' not in rsp_set:
        flink.write('run,set,part,Ft_minV,FminV,Ft_Ds,FV_Ds,Ft_Ae,FV_Ae,Ft_De,Fv_De,Ft_As,FV_As,FminD,FmaxA,FDdur,FD,FAdur,FA,FIdledur,'+
        'Lt_minV,LminV,Lt_Ds,LV_Ds,Lt_Ae,LV_Ae,Lt_De,Lv_De,Lt_As,LV_As,LminD,LmaxA,LDdur,LD,LAdur,LA,LIdledur,Ds_Delay,De_Delay,As_Delay,'+
                    'Ae_Delay,minV_Diff,AeV_Diff,D_Diff,A_Diff,magnitude,cruise_pattern,magnitude_pattern,speed_level,rate_pattern\n')
    to_keep_j=[]
    to_keep_i=[]
    j=0
    for i in range(len(oscillation_FV)):
        if (oscillation_FV[i][0] - oscillation_LV[j][0]) <= - 20:
            continue
        try:
            while (oscillation_FV[i][0] - oscillation_LV[j][0]) >= 20:
                j += 1
        except:
            break
        to_keep_i.append(i)
        to_keep_j.append(j)
        if '%s,%s,%s,'%(run,set,part) not in rsp_set:
            flink.write('%s,%s,%s,'%(run,set,part))
            for item in oscillation_FV[i]:
                flink.write('%s,'%str(item))
            for item in oscillation_LV[j]:
                flink.write('%s,'%str(item))
            flink.write('%s,' % str(max(0,oscillation_FV[i][2]-oscillation_LV[j][2])))
            flink.write('%s,' % str(max(0,oscillation_FV[i][6]-oscillation_LV[j][6])))
            flink.write('%s,' % str(max(0,oscillation_FV[i][8]-oscillation_LV[j][8])))
            flink.write('%s,' % str(max(0,oscillation_FV[i][4]-oscillation_LV[j][4])))
            flink.write('%s,' % str(round(oscillation_FV[i][1]-oscillation_LV[j][1],2)))
            flink.write('%s,' % str(round(oscillation_FV[i][5]-oscillation_LV[j][5],2)))
            flink.write('%s,' % str(round(oscillation_FV[i][13]-oscillation_LV[j][13],2)))
            flink.write('%s,' % str(round(oscillation_FV[i][15]-oscillation_LV[j][15],2)))
            flink.write('%s,' % str(round(oscillation_LV[j][3]-oscillation_LV[j][1],2)))

            #37 deceleration starting delay
            #38 deceleration ending delay
            #39 acceleration starting delay
            #40 acceleration ending delay
            #41 difference on minimum speed
            #42 difference on ending speed
            #43 Deceleration rate diff
            #44 Acceleration rate diff
            #45 LV magnitude
            if oscillation_LV[j][16] < 5:
                idle_pattern = 'dip'
            else:
                idle_pattern = 'cruise'
            flink.write('%s,' % idle_pattern)

            if round(oscillation_LV[j][3]-oscillation_LV[j][1],2) < 7.5:
                magnitude_pattern = 'mild'
            else:
                magnitude_pattern = 'strong'
            flink.write('%s,' % magnitude_pattern)

            if oscillation_LV[j][3] > 55:
                speed_group = 'high'
            elif oscillation_LV[j][3] > 40:
                speed_group = 'middle'
            else:
                speed_group = 'low'
            flink.write('%s,' % speed_group)

            if oscillation_LV[j][13] < .75:
                rate_pattern = 'mild'
            else:
                rate_pattern = 'strong'
            flink.write('%s,' % rate_pattern)

            #46 cruise pattern
            #47 magnitude pattern
            #48 speed level
            #49 deceleration/acceleration pattern

            flink.write('\n')
    flink.close()
    return [oscillation_FV[i] for i in to_keep_i],[oscillation_LV[j] for j in to_keep_j]

def traj_by_oscillation(traj, oscillation_set, extended_time, smart_extension = False):

    divided_traj=[]
    for oscillation in oscillation_set:
        if smart_extension == True:
            extended_time = oscillation[4] - oscillation[2]
        s=find_nearest_index(traj[0],max(traj[0][0],oscillation[2]-extended_time))
        e=find_nearest_index(traj[0],min(traj[0][-1],oscillation[4]+extended_time))
        divided_traj.append([i[s:e] for i in traj])
    return divided_traj


def draw_oscillation_statistics(data,x_column,y_column,label_column,x_label,mark,y_label,title='',regression=False,stick_plot=False):
    c_group={}
    c_group[mark[0]]='r'
    c_group[mark[1]]='b'
    if len(mark) == 3:
        c_group[mark[2]]='g'
    c_group['default']='k'
    data_group={}
    data_group[mark[0]]=[]
    data_group[mark[1]]=[]
    if len(mark) == 3:
        data_group[mark[2]]=[]

    if stick_plot==True:
        alpha_val=.25
    else:
        alpha_val=1

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    for d in data:
        if label_column!=-1:
            if regression:
                plt.scatter(d[x_column], d[y_column], color=c_group[d[label_column]], alpha=alpha_val)
            data_group[d[label_column]].append((d[x_column],d[y_column]))
        else:
            if regression:
                plt.scatter(d[x_column], d[y_column], color='k', alpha=1)
            data_group['default'].append((d[x_column],d[y_column]))
    data_group={k: v for k, v in data_group.items() if v!=[]}
    mean_value=[]
    for label in mark:
        data_points_x=[i[0] for i in data_group[label]]
        data_points_y=[i[1] for i in data_group[label]]
        mean_value.append((np.mean(data_points_x),#0 x location
                 np.mean(data_points_y),#1 y-mean location
                 np.percentile(data_points_y,90),#2 y-90th location
                 np.percentile(data_points_y,10),#3 y-10th location
                 np.std(data_points_y),#4 y std
                 label,#5 mode label
                 data_points_y))#6 y samples

        if regression:
            coef, intercept, p_value = linear_regression(data_points_x,data_points_y)
            if p_value<0.1:
                print(y_label,label,"p value:", p_value)
                plt.plot([min(data_points_x),max(data_points_x)],
                         [coef*min(data_points_x)+intercept,coef*max(data_points_x)+intercept],
                         color=c_group[label], alpha=1)

    if stick_plot:
        # mean_value.sort(key=lambda v: v[0])
        plt.plot([i+1 for i in range(len(mark))],[np.percentile(mv[6],50) for mv in mean_value],color='orange',linestyle='--',linewidth=1)
        box_data=[]
        for mv in mean_value:
            box_data.append(mv[6])
        plt.boxplot(box_data,whis=[5,95],labels=mark)
        # for i in range(len(mean_value)):
        #     print(y_label, int(mean_value[i][0]), 'mean', np.mean(mean_value[i][6]))
        # for i in range(len(mean_value)):
        #     print(y_label, int(mean_value[i][0]),'std' ,np.std(mean_value[i][6]))
        # for i in range(len(mean_value)):
        #     print(y_label, int(mean_value[i][0]), stats.ttest_1samp(mean_value[i][6], .0)[1])
        for i in range(len(mean_value)):
            if i>=1:
                print(y_label,int(mean_value[i][0]),int(mean_value[i-1][0]),stats.ttest_ind(mean_value[i][6],mean_value[i-1][6])[1])
        # plt.plot([mean_value[i][0],mean_value[i][0]],[mean_value[i][2],mean_value[i][3]],color='k',linewidth=2)
        # plt.plot([mean_value[i][0]-2, mean_value[i][0]+2], [mean_value[i][2], mean_value[i][2]], color='k', linewidth=2)
        # plt.plot([mean_value[i][0]-2, mean_value[i][0]+2], [mean_value[i][3], mean_value[i][3]], color='k', linewidth=2)
        # # plt.text(mean_value[i][0]-5,mean_value[i][2]*1.1,str(mean_value[i][5])+'\nmean:'+str(round(mean_value[i][1],2))+'\nstd:'+str(round(mean_value[i][4],2)))
        # print(y_label,int(mean_value[0][0]),int(mean_value[-1][0]),stats.ttest_ind(mean_value[0][6],mean_value[-1][6])[1])

    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    # plt.ylim([0,20])
    plt.savefig('figures_scatters/new/'+title+'_'+x_label[:-8]+' '+y_label[:-8]+'.png')
    plt.close()


def draw_oscillation_statistics_2(data,x_column,y_column,label_column,secondary_label_column,x_label,y_label,stick_plot=True):
    mark_label = 'cruise pattern'
    mark = ['dip', 'cruise']
    c_group={}
    c_group[mark[0]]='red'
    c_group[mark[1]]='navy'
    c_group['high']='g'
    c_group['middle']='b'
    c_group['low']='r'

    data_group={}
    data_group['high']={}
    data_group['middle']={}
    data_group['low']={}
    data_group['high'][mark[0]]=[]
    data_group['middle'][mark[0]]=[]
    data_group['low'][mark[0]]=[]
    data_group['high'][mark[1]]=[]
    data_group['middle'][mark[1]]=[]
    data_group['low'][mark[1]]=[]


    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    for d in data:
        # plt.scatter(d[x_column],d[y_column],color=c_group[d[secondary_label_column]],alpha=.25)
        data_group[d[label_column]][d[secondary_label_column]].append((d[x_column],d[y_column]))


    mean_value=[(np.mean([i[0] for i in data_group[label][mark[1]]]+[i[0] for i in data_group[label][mark[0]]]),
                 np.mean([i[1] for i in data_group[label][mark[1]]]),
                 np.percentile([i[1] for i in data_group[label][mark[1]]],90),
                 np.percentile([i[1] for i in data_group[label][mark[1]]],10),
                 np.std([i[1] for i in data_group[label][mark[1]]]),
                 label,
                 [i[1] for i in data_group[label][mark[1]]]) for label in data_group]
    mean_value.sort(key=lambda v: v[0])
    # box_data=[mv[6] for mv in mean_value]
    # plt.boxplot(box_data,whis=[5,95],labels=['Low','Middle','High'])
    line1=plt.plot([1,2,3],[mv[1] for mv in mean_value],color=c_group[mark[1]],linewidth=2)
    # plt.plot([1,2,3],[mv[2] for mv in mean_value],color='b',linewidth=2,alpha=.5,linestyle=':')
    # plt.plot([1,2,3],[mv[3] for mv in mean_value],color='b',linewidth=2,alpha=.5,linestyle=':')
    plt.scatter([1,2,3], [mv[1] for mv in mean_value], color=c_group[mark[1]])

    stick_width = .075
    stick_line_style='--'
    stick_line_width=1.5
    for i in range(len(mean_value)):
        line1_1=plt.plot([i+1,i+1],[mean_value[i][2],mean_value[i][3]],color=c_group[mark[1]],
                         linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value[i][2], mean_value[i][2]], color=c_group[mark[1]],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value[i][3], mean_value[i][3]], color=c_group[mark[1]],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        # plt.text(mean_value[i][0]-5,mean_value[i][2]*1.1,str(mean_value[i][5])+'\nmean:'+str(round(mean_value[i][1],2))+'\nstd:'+str(round(mean_value[i][4],2)))
        # if i >= 1:
        #     print(y_label,'eco', int(mean_value[i][0]), int(mean_value[i - 1][0]),
        #           stats.ttest_ind(mean_value[i][6], mean_value[i - 1][6])[1])

    mean_value_2 = [(np.mean([i[0] for i in data_group[label][mark[1]]]+[i[0] for i in data_group[label][mark[0]]]),
                   np.mean([i[1] for i in data_group[label][mark[0]]]),
                   np.percentile([i[1] for i in data_group[label][mark[0]]], 90),
                   np.percentile([i[1] for i in data_group[label][mark[0]]], 10),
                   np.std([i[1] for i in data_group[label][mark[0]]]),
                   label,
                   [i[1] for i in data_group[label][mark[0]]]) for label in data_group]
    mean_value_2.sort(key=lambda v: v[0])
    # box_data=[mv[6] for mv in mean_value_2]
    # plt.boxplot(box_data,whis=[5,95],labels=['Low','Middle','High'])
    line2=plt.plot([1,2,3], [mv[1] for mv in mean_value_2], color=c_group[mark[0]], linewidth=2)
    # plt.plot([1,2,3],[mv[2] for mv in mean_value_2],color='r',linewidth=2,alpha=.5,linestyle=':')
    # plt.plot([1,2,3],[mv[3] for mv in mean_value_2],color='r',linewidth=2,alpha=.5,linestyle=':')
    plt.scatter([1,2,3], [mv[1] for mv in mean_value_2], color=c_group[mark[0]])

    for i in range(len(mean_value_2)):
        line2_1=plt.plot([i+1,i+1], [mean_value_2[i][2], mean_value_2[i][3]], color=c_group[mark[0]],
                         linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value_2[i][2], mean_value_2[i][2]], color=c_group[mark[0]],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value_2[i][3], mean_value_2[i][3]], color=c_group[mark[0]],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        # if i >= 1:
        #     print(y_label,'power', int(mean_value_2[i][0]), int(mean_value_2[i - 1][0]),
        #           stats.ttest_ind(mean_value_2[i][6], mean_value_2[i - 1][6])[1])

    for i in range(len(mean_value_2)):
        print(y_label, int(mean_value_2[i][0]), stats.ttest_ind(mean_value[i][6], mean_value_2[i][6])[1])
        # plt.text(mean_value[i][0]-5,mean_value[i][2]*1.1,str(mean_value[i][5])+'\nmean:'+str(round(mean_value[i][1],2))+'\nstd:'+str(round(mean_value[i][4],2)))
    print(y_label, stats.ttest_ind(np.concatenate([mean_value[i][6] for i in range(len(mean_value))]),
                                   np.concatenate([mean_value_2[i][6] for i in range(len(mean_value_2))]))[1])
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.xticks((1,2,3), ('Low', 'Medium','High'))
    # plt.legend((line1[0],line1_1[0],line2[0],line2_1[0]),(mark[1],'10th/90th value',mark[0],'10th/90th value'),fontsize=10.5)
    plt.legend((line1[0], line2[0]),(mark[1], mark[0]),fontsize=10.5)

    plt.savefig('figures_scatters/new/%s_'%mark_label+y_label+'.png')
    plt.close()

def read_oscillation_data(folder_name = ''):
    group_info=get_group_info()
    data = []
    fo = open(folder_name + 'oscillation_info.csv', 'r')
    fo.readline()
    while True:
        line = fo.readline()
        if not line:
            break
        tmp = line.split(',')
        deceleration_starting_delay = float(tmp[37])
        deceleration_ending_delay = float(tmp[38])
        acceleration_starting_delay = float(tmp[39])
        acceleration_ending_delay = float(tmp[40])
        LV_starting_speed = float(tmp[23])
        minimum_speed_diff = float(tmp[41])
        end_speed_diff = float(tmp[42])
        A_diff = float(tmp[44])
        D_diff = float(tmp[43])
        FD=float(tmp[16])
        FA=float(tmp[18])
        magnitude = float(tmp[45])
        LV_D = float(tmp[33])
        LV_A = float(tmp[35])
        LV_idle_duration = float(tmp[36])
        idle_pattern = tmp[46]
        magnitude_pattern = tmp[47]
        speed_group = tmp[48]
        rate_pattern = tmp[49]

        # set = tmp[0]
        # speed_group = group_info[set][0]
        # power_mode = group_info[set][1]

        power_mode = ''
        if magnitude > 45:
            continue
        data.append([speed_group,  # 0
                     power_mode,  # 1
                     deceleration_starting_delay,  # 2
                     deceleration_ending_delay,  # 3
                     acceleration_starting_delay,  # 4
                     acceleration_ending_delay,  # 5
                     LV_starting_speed,  # 6
                     minimum_speed_diff,  # 7
                     end_speed_diff,  # 8
                     A_diff,  # 9
                     D_diff,  # 10
                     magnitude,  # 11
                     LV_A,  # 12
                     LV_D,  # 13
                     FA,#14
                     FD,#15
                     idle_pattern,#16
                     magnitude_pattern,#17
                     speed_group,#18
                     rate_pattern,#19
                     LV_idle_duration#20
                     ])
    fo.close()

    fo = open(folder_name + 'oscillation_time.csv', 'r')
    fo.readline()
    i=0
    while True:
        line = fo.readline()
        if not line:
            break
        tmp = line.split(',')
        data[i].append('headway - '+tmp[6])#21 headway setting
        i += 1
    fo.close()



    return data

def draw_oscillation_statistics_multiple_val(data,x_column,y_column_set,label_column,x_label,mark,y_label,y_limit,stick_plot=False):

    data_group={}
    data_group[mark[0]]=[]
    data_group[mark[1]]=[]
    if len(mark) == 3:
        data_group[mark[2]]=[]

    var_name={}
    var_name[2]='Deceleration start'
    var_name[4]='Acceleration start'
    var_name[5]='Acceleration end'
    var_name[12]='Leader'
    var_name[13]='Leader'
    var_name[14]='Follower'
    var_name[15]='Follower'
    c_group = {}
    c_group[12] = 'g'
    c_group[13] = 'g'
    c_group[14] = 'r'
    c_group[15] = 'r'
    c_group[2] = 'g'
    c_group[4] = 'r'
    c_group[5] = 'b'


    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    mean_value_set=[]
    for y_column in y_column_set:
        for d in data:
            if label_column != -1:
                data_group[d[label_column]].append((d[x_column], d[y_column]))
            else:
                data_group['default'].append((d[x_column], d[y_column]))
        data_group = {k: v for k, v in data_group.items() if v != []}
        mean_value = []
        for label in data_group:
            data_points_x = [i[0] for i in data_group[label]]
            data_points_y = [i[1] for i in data_group[label]]
            mean_value.append((np.mean(data_points_x),  # 0 x location
                               np.mean(data_points_y),  # 1 y-mean location
                               np.percentile(data_points_y, 90),  # 2 y-90th location
                               np.percentile(data_points_y, 10),  # 3 y-10th location
                               np.std(data_points_y),  # 4 y std
                               label,  # 5 mode label
                               data_points_y))  # 6 y samples
        # mean_value.sort(key=lambda v: v[0])
        # print(var_name[y_column],[mv[1] for mv in mean_value],[mv[6] for mv in mean_value])

        mean_value_set.append((mean_value_set,y_column))
        plt.plot(np.arange(1,len(mean_value)+1), [np.mean(mv[6]) for mv in mean_value],
                 linewidth=2,label=var_name[y_column],color=c_group[y_column],linestyle='-')
        plt.scatter(np.arange(1,len(mean_value)+1), [np.mean(mv[6]) for mv in mean_value],color=c_group[y_column])
        if stick_plot:
            stick_width = .075
            stick_line_style = '--'
            stick_line_width = 1.5
            for i in range(len(mean_value)):
                line2_1 = plt.plot([i + 1, i + 1], [mean_value[i][2], mean_value[i][3]], color=c_group[y_column],
                                   linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][2], mean_value[i][2]],
                         color=c_group[y_column],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)
                plt.plot([i + 1 - stick_width, i + 1 + stick_width], [mean_value[i][3], mean_value[i][3]],
                         color=c_group[y_column],
                         linewidth=stick_line_width, alpha=1, linestyle=stick_line_style)

    plt.xticks(np.arange(1,len(mean_value)+1), mark)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.ylim(y_limit)
    plt.legend()
    plt.savefig('figures_scatters/new/' + '_' + x_label[:-8] + ' ' + y_label[:-8]+ '_3.png')
    plt.close()




def draw_oscillation_statistics_2_and_more(data,x_column_set,x_mark,secondary_label_column,mark,x_label,y_label,stick_plot=True):

    c_group={}
    c_group[mark[0]]='r'
    c_group[mark[1]]='g'
    if len(mark) == 3:
        c_group[mark[2]]='b'

    data_group={}
    for m in x_mark:
        data_group[m]={}
        data_group[m][mark[0]]=[]
        data_group[m][mark[1]]=[]
        if len(mark) == 3:
            data_group[m][mark[2]] = []

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.15, 0.15, 0.75, 0.8])
    for d in data:
        c=0
        for x_column in x_column_set:
            data_group[x_mark[c]][d[secondary_label_column]].append((d[x_column],d[x_column]))
            c+=1
    for i in range(len(mark)):
        mean_value=[(np.mean([i[0] for i in data_group[label][mark[i]]]),
                     np.mean([i[1] for i in data_group[label][mark[i]]]),
                     np.percentile([i[1] for i in data_group[label][mark[i]]],90),
                     np.percentile([i[1] for i in data_group[label][mark[i]]],10),
                     np.std([i[1] for i in data_group[label][mark[i]]]),
                     label,
                     [i[1] for i in data_group[label][mark[i]]]) for label in data_group]
        exec('line%s=plt.plot([1,2,3,4],[mv[1] for mv in mean_value],color=c_group[mark[i]],linewidth=2)'%i)
        plt.scatter([1,2,3,4], [mv[1] for mv in mean_value], color=c_group[mark[i]])
        stick_width = .075
        stick_line_style='--'
        stick_line_width=1.5
        # for values in range(len(mean_value)):
        #     plt.plot([values+1,values+1],[mean_value[values][2],mean_value[values][3]],color=c_group[mark[i]],
        #                      linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
            # plt.plot([values+1-stick_width,values+1+stick_width], [mean_value[values][2], mean_value[values][2]],
            #          color=c_group[mark[i]],
            #          linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
            # plt.plot([values+1-stick_width,values+1+stick_width], [mean_value[values][3], mean_value[values][3]],
            #          color=c_group[mark[i]],
            #          linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)

    plt.ylabel(y_label,fontsize=16)
    plt.xticks((1,2,3,4), x_mark, fontsize=12)
    if len(mark) == 3:
        exec('plt.legend((line0[0], line1[0], line2[0]),mark,fontsize=16)')
    else:
        exec('plt.legend((line0[0], line1[0]),mark,fontsize=16)')

    plt.savefig('figures_scatters/new/%s_'%mark+y_label+'.png')
    plt.close()
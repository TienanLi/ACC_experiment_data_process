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
    v_threshold=max(max(np.nanmax(v)*0.6,np.nanmean(v)*0.8),np.nanpercentile(v,15))
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
        idle_threshold = max(1,minimum_speed[0]) * 0.02
        previous_period=v[max(0,t_p-20*expected_frequency):t_p]
        following_period=v[t_p:min(len(t),t_p+150*expected_frequency)]
        previous_d=[(previous_period[i]-previous_period[i-1])*expected_frequency for i in range(1,len(previous_period))]
        if fluent:
            moving_period=200
        else:
            moving_period=400
        previous_d=moving_average(previous_d,moving_period)
        previous_d=[previous_d[0]]+previous_d
        following_a = [(following_period[i] - following_period[i - 1])*expected_frequency for i in range(1, len(following_period))]
        following_a = moving_average(following_a, moving_period)
        following_a=following_a+[following_a[-1]]

        minimum_d=0
        start_point=len(previous_period)-1
        idle_start=len(previous_period)-1
        idle_start_speed=minimum_speed[o]
        for i in np.arange(len(previous_period)-2,1,-1):
            if (previous_period[i]-minimum_speed[o])<=idle_threshold:
                idle_start=i
                idle_start_speed=previous_period[i]
            if previous_period[i]>previous_period[start_point]:
                start_point=i
                if previous_d[i]<minimum_d:
                    minimum_d=previous_d[i]
            # elif (start_point-i)>2*expected_frequency or previous_d[i]>minimum_d*0.5 or ((previous_d[i]>(minimum_d*0.5) and previous_d[i]>-0.1) and fluent):
            elif (start_point-i)>2*expected_frequency or previous_d[i]>minimum_d*0.1:
                break

        # plt.plot(range(len(previous_period)),[p for p in previous_period])
        # plt.plot(range(len(previous_d)),[(p+2)*50 for p in previous_d])
        # plt.scatter(i,previous_d[i])
        # plt.show()

        start_speed=previous_period[start_point]
        oscillations[o].append(round(t[t_p-len(previous_period)+start_point],2))
        oscillations[o].append(round(start_speed,2))
        #2 deceleration start
        #3 deceleration start speed
        maximum_a=0
        end_point=0
        idle_end=0
        idle_end_speed=minimum_speed[o]
        for i in np.arange(1,len(following_period)):
            if (following_period[i]-minimum_speed[o])<idle_threshold:
                idle_end=i
                idle_end_speed=following_period[i]
            if following_period[i]>=following_period[end_point]:
                end_point=i
                if following_a[i]>maximum_a:
                    maximum_a=following_a[i]
            elif (following_period[i]>=start_speed*0.8) and ((i-end_point)>3*expected_frequency or following_a[i]<(maximum_a*0.05)):
                break
        end_speed=following_period[end_point]


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
        oscillations[o].append(round(minimum_d,2))
        #10 minimum deceleration rate
        oscillations[o].append(round(maximum_a,2))
        #11 maximum acceleration rate
        if min(start_speed-idle_start_speed,end_speed-idle_end_speed)<1.5 or (t_p-len(previous_period)+start_point)<1:
            oscillations.pop(o)
            minimum_speed.pop(o)
        else:
            o+=1
    for i in range(len(oscillations)):
        o=oscillations[i]
        deceleration_duration=o[6]-o[2]
        deceleration_rate=(o[3]-o[7])/deceleration_duration
        acceleration_duration=o[4]-o[8]
        acceleration_rate=(o[5]-o[9])/acceleration_duration
        idle_duration=o[8]-o[6]
        oscillations[i]=oscillations[i]+[round(deceleration_duration,2),round(deceleration_rate,2),
                                         round(acceleration_duration,2),round(acceleration_rate,2),
                                         round(idle_duration,2)]
        #12 deceleration duration
        #13 avg decleration rate
        #14 accelereation duration
        #15 avg deceleration rate
        #16 idle duration
    return oscillations

def save_oscillations(oscillation_FV,oscillation_LV,run,set,part):
    rsp_set=[]
    try:
        fo = open('oscillation_info.csv', 'r')
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
    flink = open('oscillation_info.csv', mode)
    if 'run,set,part,' not in rsp_set:
        flink.write('run,set,part,Ft_minV,FminV,Ft_Ds,FV_Ds,Ft_Ae,FV_Ae,Ft_De,Fv_De,Ft_As,FV_As,FminD,FmaxA,FDdur,FD,FAdur,FA,FIdledur,'+
        'Lt_minV,LminV,Lt_Ds,LV_Ds,Lt_Ae,LV_Ae,Lt_De,Lv_De,Lt_As,LV_As,LminD,LmaxA,LDdur,LD,LAdur,LA,LIdledur,Ds_Delay,De_Delay,As_Delay,'+
                    'Ae_Delay,minV_Diff,AeV_Diff,D_Diff,A_Diff,magnitude\n')
    to_keep_j=[]
    to_keep_i=[]
    j=0
    for i in range(len(oscillation_FV)):
        if (oscillation_FV[i][0] - oscillation_LV[j][0]) <= - 10:
            continue
        try:
            while (oscillation_FV[i][0] - oscillation_LV[j][0]) >= 10:
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
            flink.write('%s,' % str(oscillation_FV[i][2]-oscillation_LV[j][2]))
            flink.write('%s,' % str(oscillation_FV[i][6]-oscillation_LV[j][6]))
            flink.write('%s,' % str(oscillation_FV[i][8]-oscillation_LV[j][8]))
            flink.write('%s,' % str(oscillation_FV[i][4]-oscillation_LV[j][4]))
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
            flink.write('\n')
    flink.close()
    return [oscillation_FV[i] for i in to_keep_i],[oscillation_LV[j] for j in to_keep_j]

def traj_by_oscillation(traj,oscillation_set,extended_time):
    divided_traj=[]
    i=0
    for oscillation in oscillation_set:
        s=find_nearest_index(traj[0],max(traj[0][0],oscillation[2]-extended_time))
        e=find_nearest_index(traj[0],min(traj[0][-1],oscillation[4]+extended_time))
        i+=1
        divided_traj.append([i[s:e] for i in traj])
    return divided_traj


def draw_oscillation_statistics(data,x_column,y_column,label_column,x_label,y_label,title='',regression=False,stick_plot=False):
    c_group={}
    c_group['power']='r'
    c_group['eco']='b'
    c_group['high']='g'
    c_group['middle']='b'
    c_group['low']='r'
    c_group['default']='k'

    data_group={}
    data_group['high']=[]
    data_group['middle']=[]
    data_group['low']=[]
    data_group['eco']=[]
    data_group['power']=[]
    data_group['default']=[]

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
    for label in data_group:
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
        mean_value.sort(key=lambda v: v[0])
        plt.plot([1,2,3],[np.percentile(mv[6],50) for mv in mean_value],color='orange',linestyle='--',linewidth=1)
        box_data=[]
        for mv in mean_value:
            box_data.append(mv[6])
        plt.boxplot(box_data,whis=[5,95],labels=['Low','Medium','High'])
        for i in range(len(mean_value)):
            print(y_label, int(mean_value[i][0]), 'mean', np.mean(mean_value[i][6]))
        for i in range(len(mean_value)):
            print(y_label, int(mean_value[i][0]),'std' ,np.std(mean_value[i][6]))
        for i in range(len(mean_value)):
            print(y_label, int(mean_value[i][0]), stats.ttest_1samp(mean_value[i][6], .0)[1])
        for i in range(len(mean_value)):
            if i>=1:
                print(y_label,int(mean_value[i][0]),int(mean_value[i-1][0]),stats.ttest_ind(mean_value[i][6],mean_value[i-1][6])[1])
        # plt.plot([mean_value[i][0],mean_value[i][0]],[mean_value[i][2],mean_value[i][3]],color='k',linewidth=2)
        # plt.plot([mean_value[i][0]-2, mean_value[i][0]+2], [mean_value[i][2], mean_value[i][2]], color='k', linewidth=2)
        # plt.plot([mean_value[i][0]-2, mean_value[i][0]+2], [mean_value[i][3], mean_value[i][3]], color='k', linewidth=2)
        # # plt.text(mean_value[i][0]-5,mean_value[i][2]*1.1,str(mean_value[i][5])+'\nmean:'+str(round(mean_value[i][1],2))+'\nstd:'+str(round(mean_value[i][4],2)))
        print(y_label,int(mean_value[0][0]),int(mean_value[-1][0]),stats.ttest_ind(mean_value[0][6],mean_value[-1][6])[1])

    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    # plt.ylim([0,20])
    plt.savefig('figures_scatters/'+title+'_'+x_label[:-8]+' '+y_label[:-8]+'.png')
    plt.close()


def draw_oscillation_statistics_2(data,x_column,y_column,label_column,secondary_label_column,x_label,y_label,stick_plot=True):
    c_group={}
    c_group['power']='red'
    c_group['eco']='navy'
    c_group['high']='g'
    c_group['middle']='b'
    c_group['low']='r'

    data_group={}
    data_group['high']={}
    data_group['middle']={}
    data_group['low']={}
    data_group['high']['power']=[]
    data_group['middle']['power']=[]
    data_group['low']['power']=[]
    data_group['high']['eco']=[]
    data_group['middle']['eco']=[]
    data_group['low']['eco']=[]

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.15, 0.15, 0.8, 0.8])
    for d in data:
        # plt.scatter(d[x_column],d[y_column],color=c_group[d[secondary_label_column]],alpha=.25)
        data_group[d[label_column]][d[secondary_label_column]].append((d[x_column],d[y_column]))


    mean_value=[(np.mean([i[0] for i in data_group[label]['eco']]+[i[0] for i in data_group[label]['power']]),
                 np.mean([i[1] for i in data_group[label]['eco']]),
                 np.percentile([i[1] for i in data_group[label]['eco']],90),
                 np.percentile([i[1] for i in data_group[label]['eco']],10),
                 np.std([i[1] for i in data_group[label]['eco']]),
                 label,
                 [i[1] for i in data_group[label]['eco']]) for label in data_group]
    mean_value.sort(key=lambda v: v[0])
    # box_data=[mv[6] for mv in mean_value]
    # plt.boxplot(box_data,whis=[5,95],labels=['Low','Middle','High'])
    line1=plt.plot([1,2,3],[mv[1] for mv in mean_value],color=c_group['eco'],linewidth=2)
    # plt.plot([1,2,3],[mv[2] for mv in mean_value],color='b',linewidth=2,alpha=.5,linestyle=':')
    # plt.plot([1,2,3],[mv[3] for mv in mean_value],color='b',linewidth=2,alpha=.5,linestyle=':')
    plt.scatter([1,2,3], [mv[1] for mv in mean_value], color=c_group['eco'])

    stick_width = .075
    stick_line_style='--'
    stick_line_width=1.5
    for i in range(len(mean_value)):
        line1_1=plt.plot([i+1,i+1],[mean_value[i][2],mean_value[i][3]],color=c_group['eco'],
                         linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value[i][2], mean_value[i][2]], color=c_group['eco'],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value[i][3], mean_value[i][3]], color=c_group['eco'],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        # plt.text(mean_value[i][0]-5,mean_value[i][2]*1.1,str(mean_value[i][5])+'\nmean:'+str(round(mean_value[i][1],2))+'\nstd:'+str(round(mean_value[i][4],2)))
        # if i >= 1:
        #     print(y_label,'eco', int(mean_value[i][0]), int(mean_value[i - 1][0]),
        #           stats.ttest_ind(mean_value[i][6], mean_value[i - 1][6])[1])

    mean_value_2 = [(np.mean([i[0] for i in data_group[label]['eco']]+[i[0] for i in data_group[label]['power']]),
                   np.mean([i[1] for i in data_group[label]['power']]),
                   np.percentile([i[1] for i in data_group[label]['power']], 90),
                   np.percentile([i[1] for i in data_group[label]['power']], 10),
                   np.std([i[1] for i in data_group[label]['power']]),
                   label,
                   [i[1] for i in data_group[label]['power']]) for label in data_group]
    mean_value_2.sort(key=lambda v: v[0])
    # box_data=[mv[6] for mv in mean_value_2]
    # plt.boxplot(box_data,whis=[5,95],labels=['Low','Middle','High'])
    line2=plt.plot([1,2,3], [mv[1] for mv in mean_value_2], color=c_group['power'], linewidth=2)
    # plt.plot([1,2,3],[mv[2] for mv in mean_value_2],color='r',linewidth=2,alpha=.5,linestyle=':')
    # plt.plot([1,2,3],[mv[3] for mv in mean_value_2],color='r',linewidth=2,alpha=.5,linestyle=':')
    plt.scatter([1,2,3], [mv[1] for mv in mean_value_2], color=c_group['power'])

    for i in range(len(mean_value_2)):
        line2_1=plt.plot([i+1,i+1], [mean_value_2[i][2], mean_value_2[i][3]], color=c_group['power'],
                         linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value_2[i][2], mean_value_2[i][2]], color=c_group['power'],
                 linewidth=stick_line_width,alpha=1,linestyle=stick_line_style)
        plt.plot([i+1-stick_width,i+1+stick_width], [mean_value_2[i][3], mean_value_2[i][3]], color=c_group['power'],
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
    plt.legend((line1[0],line1_1[0],line2[0],line2_1[0]),('eco','10th/90th value','power','10th/90th value'),fontsize=10.5)

    plt.savefig('figures_scatters/Engine_mode_'+y_label+'.png')
    plt.close()

def read_oscillation_data():
    group_info=get_group_info()
    data = []
    fo = open('oscillation_info.csv', 'r')
    fo.readline()
    while True:
        line = fo.readline()
        if not line:
            break
        tmp = line.split(',')
        set = tmp[0]
        speed_group = group_info[set][0]
        power_mode = group_info[set][1]
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

        # if speed_group == 'middle' and LV_starting_speed < 65:
        #     continue
        if magnitude > 45:
            continue
        data.append((speed_group,  # 0
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
                     FD#15
                     ))
    fo.close()
    return data

def draw_oscillation_statistics_multiple_val(data,x_column,y_column_set,label_column,x_label,y_label,stick_plot=False):
    c_group = {}
    c_group['power'] = 'r'
    c_group['eco'] = 'b'
    c_group['high'] = 'g'
    c_group['middle'] = 'b'
    c_group['low'] = 'r'
    c_group['default'] = 'k'
    c_group[12] = 'g'
    c_group[13] = 'g'
    c_group[14] = 'r'
    c_group[15] = 'r'
    c_group[2] = 'g'
    c_group[4] = 'r'
    c_group[5] = 'b'
    for y in y_column_set:
        if y not in c_group:
            c_group[y] = c_group['default']

    data_group = {}
    data_group['high'] = []
    data_group['middle'] = []
    data_group['low'] = []
    data_group['eco'] = []
    data_group['power'] = []
    data_group['default'] = []

    var_name={}
    var_name[2]='Deceleration start'
    var_name[4]='Acceleration start'
    var_name[5]='Acceleration end'
    var_name[12]='Leader'
    var_name[13]='Leader'
    var_name[14]='Follower'
    var_name[15]='Follower'

    linestyle_name={}
    linestyle_name[2]='-'
    linestyle_name[4]='-'
    linestyle_name[5]='-'
    linestyle_name[12]='-'
    linestyle_name[14]='-'
    linestyle_name[13]='-'
    linestyle_name[15]='-'

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_position([0.2, 0.15, 0.75, 0.8])
    mean_value_set=[]
    for y_column in y_column_set:
        data_group = {}
        data_group['high'] = []
        data_group['middle'] = []
        data_group['low'] = []
        data_group['eco'] = []
        data_group['power'] = []
        data_group['default'] = []
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
        mean_value.sort(key=lambda v: v[0])
        print(var_name[y_column],[mv[1] for mv in mean_value],[mv[6] for mv in mean_value])

        mean_value_set.append((mean_value_set,y_column))
        plt.plot(np.arange(1,len(mean_value)+1), [np.mean(mv[6]) for mv in mean_value],
                 linewidth=2,label=var_name[y_column],color=c_group[y_column],linestyle=linestyle_name[y_column])
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

    plt.xticks(np.arange(1,len(mean_value)+1), ('Low', 'Medium','High'))
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.ylim([0,2])
    plt.legend()
    plt.savefig('figures_scatters/' + '_' + x_label[:-8] + ' ' + y_label[:-8]+ '_3.png')
    plt.close()


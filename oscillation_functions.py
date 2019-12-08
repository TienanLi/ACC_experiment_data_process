import numpy as np
from base_functions import moving_average,find_nearest_index

def oscillation_statistics(t,v,expected_frequency,fluent):
    v_threshold=max(min(np.nanmax(v)*0.6,np.nanmean(v)*0.8),np.nanpercentile(v,15))
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
        idle_threshold = minimum_speed[0] * 0.03
        previous_period=v[max(0,t_p-20*expected_frequency):t_p]
        following_period=v[t_p:min(len(t),t_p+150*expected_frequency)]
        previous_d=[(previous_period[i]-previous_period[i-1])/3.6*expected_frequency for i in range(1,len(previous_period))]
        if fluent:
            moving_period=200
        else:
            moving_period=400
        previous_d=moving_average(previous_d,moving_period)
        previous_d=[previous_d[0]]+previous_d
        following_a = [(following_period[i] - following_period[i - 1])/3.6*expected_frequency for i in range(1, len(following_period))]
        following_a = moving_average(following_a, moving_period)
        following_a=following_a+[following_a[-1]]

        minimum_d=0
        start_point=len(previous_period)-1
        idle_start=len(previous_period)-1
        idle_start_speed=0
        for i in np.arange(len(previous_period)-2,1,-1):
            if (previous_period[i]-minimum_speed[o])<idle_threshold:
                idle_start=i
                idle_start_speed=previous_period[i]
            if previous_period[i]>previous_period[start_point]:
                start_point=i
                if previous_d[i]<minimum_d:
                    minimum_d=previous_d[i]
            elif (start_point-i)>3*expected_frequency or previous_d[i]>0 or ((previous_d[i]>(minimum_d*0.5) and previous_d[i]>-0.1) and fluent):
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
        idle_end_speed=0
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
        if min(start_speed-idle_start_speed,end_speed-idle_end_speed)<5:
            oscillations.pop(o)
            minimum_speed.pop(o)
        else:
            o+=1
    for i in range(len(oscillations)):
        o=oscillations[i]
        deceleration_duration=o[6]-o[2]
        deceleration_rate=(o[3]-o[7])/3.6/deceleration_duration
        acceleration_duration=o[4]-o[8]
        acceleration_rate=(o[5]-o[9])/3.6/acceleration_duration
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
        'Lt_minV,LminV,Lt_Ds,LV_Ds,Lt_Ae,LV_Ae,Lt_De,Lv_De,Lt_As,LV_As,LminD,LmaxA,LDdur,LD,LAdur,LA,LIdledur,Ds_Delay,De_Delay,As_Delay,Ae_Delay,minV_Diff,AeV_Diff,\n')
    to_keep_j=[]
    to_keep_i=[]
    j=0
    for i in range(len(oscillation_FV)):
        try:
            while abs(oscillation_FV[i][0] - oscillation_LV[j][0]) >= 10:
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

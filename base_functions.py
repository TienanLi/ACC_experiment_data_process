import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
from math import sin, cos, sqrt, atan2, radians
from pyproj import Proj, transform

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
    bin_string=hex_to_byte(hex_str, total_length)[m_start:m_start + m_length]
    int_value=int(bin_string,2)
    if signed and int_value>2**(m_length-1):
        int_value=int(inverse(bin_string),2)
        int_value=~int_value
    return int_value

def inverse(string10):
    k=''
    for s in string10:
        k=k+complement(s)
    return k

def complement(inp):
    if inp=='1':
        return '0'
    if inp=='0':
        return '1'

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
            eta=tau_i * ((1 / k / w))
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
    n=int(n)
    avg_list=pd.Series(a).rolling(n, min_periods=int(n / 5)).mean().tolist()
    avg_list=avg_list[int(n/2):]+[np.nan for i in range(int(n))]
    avg_list=avg_list[:len(a)]
    return avg_list

def find_nearest_index(time_serires,point):
    series_a=[abs(ts-point) for ts in time_serires]
    return series_a.index(min(series_a))

def fill_front_space_missing_signal(serie,expected_frequency,high_threshold,unnormal=False,unnormal_down=False):
    missing_index=[]
    unnormal_threshold=250/expected_frequency

    # if serie[0]>=high_threshold:
    #     missing_index.append(0)
    for i in range(1*expected_frequency,len(serie)):
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
                    if x_start==0:
                       for m_i in missing_index:
                           serie[m_i] = interplot_end
                       missing_index=[]
                       continue
                    slope = (interplot_end - interplot_start) / (i - x_start)
                    if abs(slope)>(3/expected_frequency) and abs(interplot_start-interplot_end)>5:
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
    # if unnormal_down==True:
    #     serie=fill_front_space_missing_signal(serie,expected_frequency,high_threshold,unnormal=True)
    return serie

def get_ID_loc_and_model(run):
    messeage_ID_location=5
    model='prius'
    if run==5|7:
        model='carolla'
    if run<=3:
        model='civic'
        messeage_ID_location=1
    # print('run:',run)
    return messeage_ID_location,model

def get_group_info():
    group_info={}
    group_info['11']=('high','power')
    group_info['12']=('high','eco')
    group_info['13']=('middle','power')
    group_info['14']=('middle','eco')
    group_info['15']=('low','power')
    group_info['16']=('low','eco')
    group_info['17']=('middle','power')
    return group_info

def get_speed_range(run):
    group_info=get_group_info()
    if group_info[run][0]=='high':
        speed_range=[10,30]
    elif group_info[run][0] == 'middle':
        speed_range=[8,20]
    else:
        speed_range=[0,8]
    return speed_range


def divide_traj(traj,expected_frequency,period_length):
    period_length=period_length*expected_frequency
    if len(traj[0])<=period_length:
        return [traj]
    divided_traj=[]
    i=0
    while True:
        s=i*int(period_length*0.8)
        e=s+period_length
        if e>(len(traj[0])-period_length):
            divided_traj.append([i[s:] for i in traj])
            break
        else:
            divided_traj.append([i[s:e] for i in traj])
        i+=1

    return divided_traj


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

def linear_regression(X,Y):
    X=np.array(X).reshape(len(X),1)
    Y=np.array(Y).reshape(len(Y),1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    y_pred = regr.predict(X)

    params = np.append(regr.intercept_, regr.coef_)
    params = np.round(params, 2)

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((Y-y_pred)**2))/(len(newX)-len(newX[0]))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    return params[1],params[0],p_values[1]



def cal_distance(point_1,point_2):
    # approximate radius of earth in m
    R = 6373.0
    lat1 = radians(point_1[0])
    lon1 = radians(point_1[1])
    lat2 = radians(point_2[0])
    lon2 = radians(point_2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c * 1000
    return distance
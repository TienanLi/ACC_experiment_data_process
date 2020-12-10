import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
from sklearn import linear_model, metrics
from scipy import stats
from math import sin, cos, sqrt, atan2, radians, floor, log10
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
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
            closest_j = y_list.index(min(y_list))
            leader_t = closest_j * sim_freq + t[r[0]]

            closest_j_on_t = closest_j + r[0]
            closest_y_diff = (-w*(t[closest_j_on_t]-t_f[i])+d_f[i])-d[closest_j_on_t]
            if closest_y_diff >= 0:
                if closest_j_on_t < len(d):
                    v = (d[closest_j_on_t + 1] - d[closest_j_on_t]) / 0.1
                else:
                    v = (d[closest_j_on_t] - d[closest_j_on_t - 1]) / 0.1

                interplot = closest_y_diff / (w + v)
                leader_t = leader_t + interplot
            if closest_y_diff < 0:
                if closest_j_on_t == 0:
                    v = (d[closest_j_on_t + 1] - d[closest_j_on_t]) / 0.1
                else:
                    v = (d[closest_j_on_t] - d[closest_j_on_t - 1]) / 0.1
                interplot = - closest_y_diff / (w + v)
                leader_t = leader_t - interplot

            #use t ratio
            tau_i=t_f[i]-leader_t
            eta=tau_i / ((1 / k / w))
            # eta = tau_i

            #use d ratio
            # d_i = d[find_nearest_index(t,leader_t)] - d_f[i]
            # eta = d_i / ((1 / k ))

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

def cal_ita_dynamic_wave_fix_tau0(t,d,t_f,d_f,tau0,stand_still_spacing):
    ita=[]
    t_ita=[]
    for i in range(len(t_f)):
        try:
            lead_position_t_before = d[find_nearest_index(t,t_f[i]-tau0)]
            my_position = d_f[i]
            # eta = lead_position_t_before - my_position
            eta = (lead_position_t_before - my_position) / stand_still_spacing
            # if eta<0:
            #     eta=np.nan
            ita.append(eta)
            t_ita.append(t_f[i])
        except:
            break
    return t_ita,ita


def moving_average(a, n) :
    n=int(n)
    avg_list=pd.Series(a).rolling(n, min_periods=int(n / 5)).mean().tolist()
    avg_list=avg_list[int(n/2):]+[np.mean(a[i:]) for i in range(-n, 0)]
    avg_list=avg_list[:len(a)]
    return avg_list

def find_nearest_index(time_serires,point):
    series_a=[abs(ts-point) for ts in time_serires]
    return series_a.index(np.nanmin(series_a))

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

def linear_regression(X, Y, weight = None):
    X=np.array(X).reshape(len(X),1).astype(np.float32)
    Y=np.array(Y).reshape(len(Y),1).astype(np.float32)
    print('size:',len(X))
    regr = linear_model.LinearRegression()
    regr.fit(X, Y, sample_weight = weight)
    y_pred = regr.predict(X)

    print('R2=',round(metrics.r2_score(Y, y_pred),3))
    params = np.append(regr.intercept_, regr.coef_)
    params = np.array([round_sig(p, sig = 2) for p in params])

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((Y-y_pred)**2))/(len(newX)-len(newX[0]))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    print('p_value intercept:', round(p_values[0],3), 'coef:', round(p_values[1],3))
    return params[1],params[0],p_values[1]

def multi_linear_regression(X, y, weight = None, X_index = None):
    print('size:',len(X))
    lm = linear_model.LinearRegression()
    lm.fit(X, y)
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)
    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)
    myDF3 = pd.DataFrame()
    if X_index is not None:
        myDF3['Variable']=['intercept']+X_index
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["p values"] = \
        [params, sd_b, ts_b,p_values]
    # print(myDF3)
    return myDF3

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


def get_a_part_before_a_point(data, time_data, stab_point, duration):
    return data[find_nearest_index(time_data, stab_point-duration):find_nearest_index(time_data, stab_point)]

def oblique(data):
    slope = (data[-1] - data[0]) / (len(data) - 1)
    oblique_data = [data[j] - data[0] - slope * j for j in range(len(data))]
    return oblique_data

def get_stable_speed(speed):
    stable_speed = [35, 45, 65]
    if speed > 61:
        return stable_speed[2]
    elif speed > 42:
        return stable_speed[1]
    else:
        return stable_speed[0]


def WT_MEXH(y, frequency_bound = 32, prominence = 1):
    coef, freqs = pywt.cwt(y, np.arange(1, frequency_bound + 1), "mexh")
    z = 0
    h = np.zeros(len(y))
    while z < frequency_bound:
        h += np.power(coef[z], 2)
        z += 1
    total_energy = h / frequency_bound
    peak_wt = find_peaks(total_energy, prominence=prominence)[0]  # use prominence or width

    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # plt.plot(y)
    # ax = fig.add_subplot(212)
    # plt.plot(total_energy)
    # plt.show()

    return peak_wt, total_energy

def time_of_week_to_hms(time_point, time_zone):
    hour = np.floor(time_point / (60 * 60) % 24) + time_zone
    minute = np.floor(round(time_point / (60 * 60) % 1 * 60)) #use round to avoid numerical error from the division
    sec = np.floor(round(time_point / (60 * 60) % 1 * 60) % 1 * 60)
    return hour, minute, sec

def exclude_outlier(data, split = 5, exclude_threshold = 1):
    X = data[0]
    slots = np.arange(min(X), max(X) + 1, (max(X) - min(X)) / split)
    # slots = [0,33,37,53,57,63,67,70]
    included_data = pd.DataFrame()
    for i in range(len(slots) - 1):
        selected = data[(data[0] >= slots[i]) & (data[0] <= slots[i + 1])]
        y_mean = np.mean(selected[1])
        y_std = np.std(selected[1])
        upper_bound = min(y_mean + y_std * exclude_threshold, 100)
        lower_bound = max(y_mean - y_std * exclude_threshold, 5)
        selected = selected[(selected[1] >= lower_bound) & (selected[1] <= upper_bound)]
        included_data = pd.concat([included_data, selected])
    # included_data = data
    return included_data

def threeD_contour_interploted(X,Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    a, b = np.mgrid[min(X[:,0]):max(X[:,0]):10j, min(X[:,1]):max(X[:,1]):10j]
    z = griddata(X, Y, (a, b), method='linear')
    z = np.array(z)
    z = z.reshape((len(a), len(b)))
    ax.plot_surface(a,b,z)

def round_sig(x, sig=2):
    digit = len(str(abs(int(x))))
    sig = digit + sig
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def assign_weight(data, slots):
    weight = []
    for i in range(len(slots) - 1):
        upper_bound = slots[i + 1]
        lower_bound = slots[i]
        if i == 0:
            selected = data[(data >= lower_bound) & (data <= upper_bound)]
        else:
            selected = data[(data > lower_bound) & (data <= upper_bound)]
        count_num = len(selected)
        print('range:',slots[i],slots[i+1],'sample:',len(selected))
        if count_num > 0:
            weight += [1/count_num for j in range(count_num)]

    return weight

def min_speed_fine_measurment(oscillations_LV, oscillations_FV, v_LV, v_FV, time_axis):
    LV_cruise = v_LV[find_nearest_index(time_axis, oscillations_LV[0][6]):
                     find_nearest_index(time_axis, oscillations_FV[0][8])]
    FV_cruise = v_FV[find_nearest_index(time_axis, oscillations_LV[0][6]):
                     find_nearest_index(time_axis, oscillations_FV[0][8])]
    time_cruise = time_axis[find_nearest_index(time_axis, oscillations_LV[0][6]):
                     find_nearest_index(time_axis, oscillations_FV[0][8])]

    for i in range(1, len(LV_cruise)):
        speed_diff = LV_cruise[i] - FV_cruise[i]
        early_speed_diff = LV_cruise[i-1] - FV_cruise[i-1]
        if (speed_diff <= 0) and (early_speed_diff >= 0):
            break
    LV_min = min(LV_cruise[:i])
    FV_min = min(FV_cruise[:i])
    LV_min_time = time_cruise[np.argmin(LV_cruise[:i])]
    FV_min_time = time_cruise[np.argmin(FV_cruise[:i])]
    oscillations_LV[0][0] = LV_min_time
    oscillations_LV[0][1] = LV_min
    oscillations_FV[0][0] = FV_min_time
    oscillations_FV[0][1] = FV_min

    return oscillations_LV, oscillations_FV

def min_speed_fine_measurment_2(oscillations_LV, oscillations_FV, v_LV, time_axis):
    LV_cruise = v_LV[find_nearest_index(time_axis, oscillations_FV[0][6]):
                     find_nearest_index(time_axis, oscillations_FV[0][8])]
    time_cruise = time_axis[find_nearest_index(time_axis, oscillations_FV[0][6]):
                     find_nearest_index(time_axis, oscillations_FV[0][8])]

    LV_min = min(LV_cruise)
    LV_min_time = time_cruise[np.argmin(LV_cruise)]
    oscillations_LV[0][0] = LV_min_time
    oscillations_LV[0][1] = LV_min

    return oscillations_LV


def derivative(X, frequency = 0.1): #n to n-1 dimension
    X = np.array(X)
    diff = X[1:] - X[:-1]
    derivative = diff / frequency
    return derivative

def denoise_speed(X, derThreshold = (-5, 5)):
    X = np.array(X)
    X = X * 0.44704 #from mph to m/s
    Xprime = derivative(X)
    noises = Xprime[(Xprime > derThreshold[1]) | (Xprime < derThreshold[0])]
    while len(noises) > 0:
        for i in range(len(Xprime) - 1):
            if (Xprime[i] > derThreshold[1]) or (Xprime[i] < derThreshold[0]):
                X[i + 1] = (X[i] + X[i + 2]) / 2
        if (Xprime[-1] > derThreshold[1]) or (Xprime[-1] < derThreshold[0]):
            X[-1] = 2 * X[-2] - X[-3]
        Xprime = derivative(X)
        noises = Xprime[(Xprime > derThreshold[1]) | (Xprime < derThreshold[0])]
    return X / 0.44704 #from m/s back to mph

def denoise_loc(loc, speedRef, errRange = 3):
    resolution = 0.1
    speedRef = np.array(speedRef)
    speedRef = speedRef * 0.44704 #from mph to m/s
    locPrime = derivative(loc, frequency=resolution)
    noises = 1
    iteration = 0
    while noises > 0:
        iteration += 1
        noises = 0
        i = 0
        while i < len(locPrime) - 1:
            if (locPrime[i] > speedRef[i] + errRange) or (locPrime[i] < speedRef[i] - errRange):
                iStart, iEnd = i, i
                if locPrime[i] > speedRef[i] + errRange:
                    sign = 1
                if locPrime[i] < speedRef[i] - errRange:
                    sign = -1
                noises += 1
                while i < len(locPrime) - 1:
                    i += 1
                    if sign == 1:
                        if locPrime[i] < speedRef[i] - errRange:
                            iEnd = i
                    if sign == -1:
                        if locPrime[i] > speedRef[i] + errRange:
                            iEnd = i

                    if iEnd > iStart:
                        if iStart > 0:
                            slope = (loc[iEnd + 1] - loc[iStart]) / (iEnd - iStart + 1)
                            for j in range(iStart + 1, iEnd + 1):
                                loc[j] = loc[j - 1] + slope
                        else:
                            for j in range(iEnd, iStart - 1, -1):
                                loc[j] = loc[j + 1] - speedRef[j] * resolution
                        break
                    if i > iStart + 10:
                        break
                if iStart == iEnd:
                    if iStart == 0:
                        loc[iStart] = 2 * loc[iStart + 1] - loc[iStart + 2]
                    else:
                        loc[iStart + 1] = (loc[iStart] + loc[iStart + 2]) / 2
            i += 1

        if (locPrime[-1] > speedRef[-1] + errRange) or (locPrime[-1] < speedRef[-1] - errRange):
            loc[-1] = 2 * loc[-2] - loc[-3]
            noises += 1
        locPrime = derivative(loc, frequency=resolution)
        if iteration > 10:
            break
    return loc

# def denoise_loc(loc, derThreshold = (-5, 5)):
#     loc = np.array(loc)
#     speedPrime = derivative(derivative(loc))
#     noises = 1
#     iteration = 0
#     while noises > 0:
#         iteration += 1
#         noises = 0
#         i = 0
#         while i < len(speedPrime):
#             if (speedPrime[i] > derThreshold[1]) or (speedPrime[i] < derThreshold[0]):
#                 iStart, iEnd = i, i
#                 if speedPrime[i] > derThreshold[1]:
#                     sign = 1
#                 if speedPrime[i] < derThreshold[0]:
#                     sign = -1
#                 noises += 1
#                 while i < len(speedPrime) - 1:
#                     i += 1
#                     if sign == 1:
#                         if speedPrime[i] < derThreshold[0]:
#                             iEnd = i
#                     if sign == -1:
#                         if speedPrime[i] > derThreshold[1]:
#                             iEnd = i
#
#                     if iEnd > iStart:
#                         if iStart > 0:
#                             slope = (loc[iEnd + 1] - loc[iStart]) / (iEnd - iStart + 1)
#                             for j in range(iStart + 1, iEnd + 1):
#                                 loc[j] = loc[j - 1] + slope
#                         else:
#                             for j in range(iEnd, iStart - 1, -1):
#                                 loc[j] = 2 * loc[j + 1] - loc[j + 2]
#                         break
#                     if i > iStart + 10:
#                         break
#                 if iStart == iEnd:
#                     if iStart == 0:
#                         loc[iStart] = 2 * loc[iStart + 1] - loc[iStart + 2]
#                     else:
#                         loc[iStart + 1] = (loc[iStart] + loc[iStart + 2]) / 2
#             i += 1
#
#         speedPrime = derivative(derivative(loc))
#         if iteration > 10:
#             break
#     return loc
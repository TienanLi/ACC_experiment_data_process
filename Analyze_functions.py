import matplotlib.pyplot as plt
import os
import numpy as np
from civic import analyze_ENGINE_DATA,analyze_KINEMATICS
from prius import analyze_SPEED,analyze_PCM_CRUISE,analyze_LEAD_INFO
from base_functions import draw_fig,cal_ita,moving_average,find_nearest_index,fill_front_space_missing_signal,\
    ACC_in_use,get_speed_range,divide_traj
from eta_functions import eta_pattern,best_ita_parameter
from oscillation_functions import oscillation_statistics,save_oscillations,traj_by_oscillation
from matplotlib.collections import LineCollection
import matplotlib.cm as pcm
from matplotlib import rc

global expected_frequency
expected_frequency = 100

font = {'family': 'DejaVu Sans',
        'size': 14}
rc('font', **font)

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


def analyze_and_draw(messeage_dict,model,run,set):
    [speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using]=analyze_CANBUS(messeage_dict,model)
    # traj_info=(speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using)
    traj_info=ACC_in_use(speed_time_series, speed, LEAD_INFO_time_series, front_space, relative_speed, ACC_using_ts, ACC_using)
    part=1
    for traj in traj_info:
        t, v, d, v_LV_derived, d_LV, t_ita, ita=traj_derivation(traj)
        # continue
        oscillations_LV=oscillation_statistics(t,v_LV_derived,expected_frequency,fluent=False)
        oscillations_FV=oscillation_statistics(t,v,expected_frequency,fluent=True)
        # oscillations_FV,oscillations_LV=save_oscillations(oscillations_FV,oscillations_LV,run,set,part)
        print(run, set, part)
        # divided_traj=divide_traj([t, v, d, v_LV_derived, d_LV, t_ita, ita],expected_frequency,period_length=50)
        divided_traj=traj_by_oscillation([t, v, d, v_LV_derived, d_LV, t_ita, ita],oscillations_FV,extended_time=20)
        split=1
        for (t, v, d, v_LV_derived, d_LV, t_ita, ita) in divided_traj:
            save_traj_info(t, v, d, v_LV_derived, d_LV, run, set, part, split)
            try:
                os.stat('figures/' + str(run) +'/')
            except:
                os.mkdir('figures/' + str(run) +'/')
            draw_traj(t, v, d, v_LV_derived, d_LV, t_ita, ita,oscillations_FV,oscillations_LV,
                      'figures/' + str(run) +'/'+str(run)+'_' + str(set) + '_part' + str(part)+'_oscillation'+str(split),run,set,split)
            split+=1
        part += 1


def traj_derivation(traj):
    moving_window=2
    veh_length=5
    (speed_time, speed, front_space_time, front_space, relative_speed)=traj
    v = [s/3.6 for s in speed]#m/s
    original_location = [0]
    for i in range(len(speed) - 1):
        forward = (speed_time[i + 1] - speed_time[i]) * v[i]
        original_location.append(original_location[-1] + forward)  # in meter
    t = speed_time
    d = original_location
    draw_fig(t, '', front_space, 'space(m)')
    front_space = fill_front_space_missing_signal(front_space,expected_frequency, high_threshold=100)
    space = front_space
    # r_v=relative_speed
    draw_fig(t, '', space, 'revised space (m)')
    d_LV = [d[i] + space[i] + veh_length for i in range(len(d))]
    # best_ita_parameter(t, d_LV, t, d, sim_freq=1/expected_frequency)
    # d_LV=moving_average(d_LV,100)
    # d_LV_derived=[d[0]+space[0]]
    # for i in range(len(d)-1):
    #     d_LV_derived.append(d_LV_derived[i]+(v[i]+r_v[i])/3.6*0.01)
    # diff=np.mean(d_LV)-np.mean(d_LV_derived)
    # d_LV_derived=[d+diff for d in d_LV_derived]
    t_ita, ita = cal_ita(t, d_LV, t, d, sim_freq=1/expected_frequency, w=5, k=.2)
    # t_ita_derived,ita_derived=cal_ita(t,d_LV_derived,t,d,sim_freq=0.01,w=5,k=0.1333)
    # v_LV_measured=[v[i]+r_v[i] for i in range(len(v))]
    v_LV_derived = [(d_LV[i + 1] - d_LV[i - 1]) * expected_frequency / 2 for i in range(1, len(d_LV) - 1)]
    v_LV_derived = [v_LV_derived[0]] + v_LV_derived + [v_LV_derived[-1]]
    v_LV_derived = moving_average(v_LV_derived, moving_window*expected_frequency)
    v_LV_derived = [max(vlv,0) for vlv in v_LV_derived]
    return t,v,d,v_LV_derived,d_LV,t_ita,ita

def draw_traj(t,v,d,v_LV_derived,d_LV,t_ita,ita,oscillations_FV,oscillationS_LV,fig_name,run,set,split):
    # [t0s_i, t0e_i, t_min_i, t_max_i, t1s_i, t1e_i, tau0, tau_min, tau_max, tau1, ep0, ep1, ep2] = \
    #     eta_pattern(t, ita, oscillations_FV[split-1], expected_frequency)
    t_p = t[0]
    t_p_max = t[-1]
    regression = False
    # if tau_max>0.75:
    #     t_p=t[t_min_i]
    #     t_p_max=t[t_max_i]
    #     regression=True
    # if min(t1e_i-t1s_i,t0e_i-t0s_i)>2*expected_frequency:
    # print(run, set, split, round(tau0, 2), round(tau_min, 2), round(tau_max, 2), round(tau1, 2), int(ep0), int(ep1), int(ep2))

    speed_range=get_speed_range(run)
    fig = plt.figure(figsize=(8, 12), dpi=300)
    ax = fig.add_subplot(311)
    ax.set_position([0.15, 0.7, 0.82, 0.25])
    color_indicator = np.array(v)
    points = np.array([np.array(t), np.array(d)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(speed_range[0], speed_range[1])
    lc = LineCollection(segments, cmap='jet_r', norm=norm)
    lc.set_array(color_indicator)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)
    color_indicator = np.array(v_LV_derived)
    points = np.array([np.array(t), np.array(d_LV)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(speed_range[0], speed_range[1])
    lc = LineCollection(segments, cmap='jet_r', norm=norm)
    lc.set_array(color_indicator)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)
    plt.ylabel('location(m)', fontsize=16)
    plt.ylim([min(d), max(d_LV)])
    plt.xlim(t[0]+3,t[-1])
    plt.title(fig_name,fontsize=16)
    ax.locator_params(nbins=5, axis='x')

    # plt.xlim([t[t0s_i], t[t1e_i]])

    cmap_jet = pcm.get_cmap('jet_r')
    sm = plt.cm.ScalarMappable(cmap=cmap_jet, norm=plt.Normalize(vmin=speed_range[0], vmax=speed_range[1]))
    cbar = plt.colorbar(sm, orientation='horizontal', cax=plt.axes([0.2, 0.65, 0.65, 0.025]))
    cbar.set_label('speed (m/s)', fontsize=16)

    bx = fig.add_subplot(312)
    bx.set_position([0.15, 0.35, 0.82, 0.25])
    ita_range=[0,3]
    plt.plot(t_ita, ita, color='g',label='direct measured from radar')
    # if regression:
    #     plt.plot([t[t0s_i], t[t0e_i], t_p, t_p_max, t[t1s_i], t[t1e_i]], [tau0, tau0, tau_min, tau_max, tau1, tau1], color='k', linewidth=2)
    #     plt.text(t[t0s_i+400], 1.3, '$\eta^0:$' + str(round(tau0, 2)) + '   $\eta^{min}:$' + str(round(tau_min, 2))
    #              + '   $\eta^{max}:$' + str(round(tau_max, 2))+ ' $\eta^1:$' + str(round(tau1, 2)),
    #              fontsize=16)
    #     plt.plot([t[t_min_i], t[t_min_i]], ita_range, color='k', linestyle='--', linewidth=1, alpha=.5)
    #     plt.plot([t[t_max_i], t[t_max_i]], ita_range, color='k', linestyle='--', linewidth=1, alpha=.5)
    plt.ylabel(r'$\tau$', fontsize=16)
    plt.xlim(t[0]+3,t[-1])
    bx.locator_params(nbins=5, axis='x')

    # plt.xlim([t[t0s_i], t[t1e_i]])
    plt.ylim(ita_range)

    cx = fig.add_subplot(313)
    cx.set_position([0.15, 0.075, 0.82, 0.225])
    plt.plot(t, v, color='r', label='Follower')
    plt.plot(t, v_LV_derived, color='g', label='Leader')
    t_shift = [t[x] + 1.87 * ita[x] for x in range(len(t))]
    # plt.plot(t_shift, v_LV_derived, color='b', linestyle='--', label='shifted LV')
    cx.locator_params(nbins=5, axis='x')

    # if regression:
    #     plt.plot([t[t_min_i], t[t_min_i]], speed_range, color='k', linestyle='--', linewidth=1, alpha=.5)
    #     plt.plot([t[t_max_i], t[t_max_i]], speed_range, color='k', linestyle='--', linewidth=1, alpha=.5)
    # plt.plot(t, v_LV_measured, color='k', label='LV (direct measured from radar)')
    for o in oscillations_FV:
        plt.scatter(o[6],o[7],color='r',s=60)
        plt.scatter(o[8],o[9],color='r',s=60)
        plt.scatter(o[2],o[3],color='r',s=60)
        plt.scatter(o[4],o[5],color='r',s=60)
        # plt.text(o[2],o[3],str(o[12])+'s\nd=-'+str(o[13])+'$m/s^2$')
        # plt.text(o[4],o[5],str(o[14])+'s\na='+str(o[15])+'$m/s^2$')
        # plt.text(o[6],o[7],str(o[16])+'s')
        plt.scatter(o[0],o[1],color='k',marker='*',s=60)
    for o in oscillationS_LV:
        plt.scatter(o[6],o[7],color='g',s=60)
        plt.scatter(o[8],o[9],color='g',s=60)
        plt.scatter(o[2],o[3],color='g',s=60)
        plt.scatter(o[4],o[5],color='g',s=60)
        plt.scatter(o[0],o[1],color='k',marker='*',s=60)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('speed(m/s)', fontsize=20)
    plt.legend(loc=4,fontsize=16)
    plt.xlim(t[0]+3,t[-1])

    # plt.xlim([t[t0s_i], t[t1e_i]])
    plt.ylim(speed_range)
    plt.savefig(fig_name + '.png')
    plt.close()

def save_traj_info(t, v, d, v_LV_derived, d_LV,run,set,part,sub,period=None):
    if period==None:
        # t=[tt-t[0] for tt in t]
        flink = open('traj_output/run_%s_set_%s_part_%s_oscillation_%s.csv'%(run,set,part,sub),'w')
        flink.write('time stamp(sec),follower location(m),follower speed(m/s),leader location(m),leader speed(m/s)\n')
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

def analyze_CANBUS(messeage_dict,model):
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



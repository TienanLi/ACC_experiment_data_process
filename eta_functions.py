from base_functions import find_nearest_index
import numpy as np


def best_ita_parameter(t_lv, d_lv, t_sv, d_sv, sim_freq):
    best_MSE=(1e8,3,0.133)
    step=1
    round_num=2
    while step>0.01:
        k_step = step / 10
        w_range=np.arange(best_MSE[1] - step * round_num, best_MSE[1] + step * (round_num+1), step)
        k_range=np.arange(best_MSE[2]-k_step*round_num,best_MSE[2]+k_step*(round_num+1),k_step)
        k_range=k_range[:round_num*2+1]
        w_range=[w for w in w_range if w>0]
        k_range=[k for k in k_range if k>0]
        # print(w_range,k_range,step)
        for w in w_range:
            for k in k_range:
                tau_0=w*k
                d=1/k
                shift_d=[dd_lv-d for dd_lv in d_lv]
                shift_t_lv=[tt_lv + tau_0 for tt_lv in t_lv]
                start_i=find_nearest_index(t_sv,shift_t_lv[0])
                SE_d=[(shift_d[i-start_i]-d_sv[i])**2 for i in range(start_i,len(t_sv))]
                # t_eta,eta=cal_ita(t_lv, d_lv, t_sv, d_sv, sim_freq, w, k)
                # eta_mean=np.nanmean(eta)
                # SE_d=[((e-eta_mean)/k)**2 for e in eta if e>0]
                MSE=np.mean(SE_d)
                # print(MSE,w,k)
                if MSE<best_MSE[0] and (len(SE_d)>0.9*len(t_lv)):
                    best_MSE=(MSE,w,k)
        # print(best_MSE)
        if len(w_range)<=2 or ((best_MSE[1]!=w_range[0]) and (best_MSE[1]!=w_range[-1])):
            if len(k_range)<=2 or ((best_MSE[2] != k_range[0]) and (best_MSE[2]!=k_range[-1])):
                step=step/2
    print(round(best_MSE[1],3),round(best_MSE[2],3),len(t_sv))

    def eta_pattern(t, tau, oscillation, expected_frequency):

        start = find_nearest_index(t, oscillation[2])
        end = find_nearest_index(t, oscillation[4])
        tau_max = find_nearest_index(t, oscillation[0])

        if tau[tau_max] < tau[start] and tau[tau_max] < tau[end]:
            period_tau = tau[start:end]

            # tau_min
            min_index = [i for i, x in enumerate(period_tau) if x == min(period_tau)]
            t_p = start + int(np.mean(min_index))
            tau_min = tau[t_p]
            previous_period = tau[max(0, t_p - 20 * expected_frequency):t_p]
            following_period = tau[t_p:min(len(t), t_p + 150 * expected_frequency)]

            # tau_0
            start_point = len(previous_period) - 1
            for i in np.arange(len(previous_period) - 2, 1, -1):
                if previous_period[i] >= previous_period[start_point]:
                    start_point = i
                elif (start_point - i) > 2 * expected_frequency:
                    break
            start_tau = previous_period[start_point]
            stable_threshold = 0.05
            start_tau_start_point = t_p - len(previous_period) + start_point
            for i in range(start_tau_start_point, -1, -1):
                if abs(tau[start_tau_start_point] - tau[i]) > stable_threshold / 1.5:
                    break
            start_tau_start_point = i

            start_tau_end_point = t_p - len(previous_period) + start_point
            for i in range(start_tau_end_point, len(t)):
                if abs(tau[start_tau_end_point] - tau[i]) > stable_threshold / 1.5:
                    break
            start_tau_end_point = i

            # tau_max
            # max_point = 0
            # for i in np.arange(1, len(following_period)):
            #     if following_period[i] >= following_period[max_point]:
            #         max_point = i
            #     elif (i - max_point) > 2 * expected_frequency:
            #         break
            # t_max=max_point+t_p
            # tau_max = following_period[max_point]

            max_index = [i for i, x in enumerate(period_tau) if x > (max(period_tau) - stable_threshold / 1.5)]
            t_max = start + int(np.mean(max_index))
            max_point = t_max - t_p
            tau_max = tau[t_max]

            # tau_1
            end_point = max_point
            for i in np.arange(end_point, len(following_period)):
                if following_period[i] <= following_period[end_point]:
                    end_point = i
                elif (i - end_point) > 2 * expected_frequency:
                    break
            # end_tau = following_period[end_point]

            end_tau_start_point = t_p + end_point
            for i in range(end_tau_start_point, -1, -1):
                if abs(tau[end_tau_start_point] - tau[i]) > stable_threshold / 1.5:
                    break
            end_tau_start_point = i

            end_tau_end_point = t_p + end_point
            for i in range(end_tau_end_point, len(t)):
                if abs(tau[end_tau_end_point] - tau[i]) > stable_threshold / 1.5:
                    break
            end_tau_end_point = i

            if abs(tau_max - tau_min) < 0.01:
                return [0 for i in range(13)]

            # best_fit=1e8
            # best_i=start_tau_end_point
            # for i in range(start_tau_end_point,start_tau_start_point,-1):
            #     slope=(tau[i]-tau_min)/(t_p-i)
            #     fit=np.nanmean([(tau[ii]-(tau[i]-(ii-i)*slope))**2 for ii in range(i,t_p)])
            #     if fit<best_fit:
            #         best_fit=fit
            #         best_i=i
            # start_tau_end_point=best_i
            #
            # best_fit = 1e8
            # best_i = end_tau_start_point
            # for i in range(end_tau_start_point,end_tau_end_point):
            #     slope=(tau_max-tau[i])/(i-max_point)
            #     fit = np.nanmean([(tau[ii] - ((i-ii) * slope + tau[i])) ** 2 for ii in range(max_point, i)])
            #     if fit < best_fit:
            #         best_fit = fit
            #         best_i = i
            # end_tau_start_point = best_i

            start_tau_start_point = start_tau_end_point
            for i in range(start_tau_end_point, -1, -1):
                if abs(tau[start_tau_end_point] - tau[i]) <= stable_threshold:
                    start_tau_start_point = i
                else:
                    break
            end_tau_end_point = end_tau_start_point
            for i in range(end_tau_start_point, len(t)):
                if abs(tau[end_tau_start_point] - tau[i]) <= stable_threshold:
                    end_tau_end_point = i
                else:
                    break

            tau0 = np.nanmean(tau[start_tau_start_point:start_tau_end_point])
            tau1 = np.nanmean(tau[end_tau_start_point:end_tau_end_point])
            slope_0 = - (tau_min - tau[start_tau_end_point]) / (t_p - start_tau_end_point) * expected_frequency * 3600
            slope_1 = (tau_max - tau_min) / (t_max - t_p) * expected_frequency * 3600
            slope_2 = (tau_max - tau[end_tau_start_point]) / (end_tau_start_point - t_max) * expected_frequency * 3600

            # if min(slope_0,slope_1)<75:
            #     return 0, 0, 0, 0, 0, 0,0,0,0,0

            return [start_tau_start_point, start_tau_end_point, t_p, t_max, end_tau_start_point, end_tau_end_point,
                    tau0, tau_min, tau_max, tau1, slope_0, slope_1, slope_2]
        return [0 for i in range(13)]


def eta_pattern(t,tau,oscillation,expected_frequency):

    start=find_nearest_index(t,oscillation[2])
    end=find_nearest_index(t,oscillation[4])
    tau_max=find_nearest_index(t,oscillation[0])

    if tau[tau_max]<tau[start] and tau[tau_max]<tau[end]:
        period_tau=tau[start:end]

        #tau_min
        min_index=[i for i, x in enumerate(period_tau) if x == min(period_tau)]
        t_p=start+int(np.mean(min_index))
        tau_min=tau[t_p]
        previous_period = tau[max(0, t_p - 20 * expected_frequency):t_p]
        following_period = tau[t_p:min(len(t), t_p + 150 * expected_frequency)]

        #tau_0
        start_point = len(previous_period) - 1
        for i in np.arange(len(previous_period) - 2, 1, -1):
            if previous_period[i] >= previous_period[start_point]:
                start_point = i
            elif (start_point - i) > 2 * expected_frequency:
                break
        start_tau = previous_period[start_point]
        stable_threshold=0.05
        start_tau_start_point=t_p-len(previous_period)+start_point
        for i in range(start_tau_start_point,-1,-1):
            if abs(tau[start_tau_start_point]-tau[i])>stable_threshold/1.5:
                break
        start_tau_start_point = i

        start_tau_end_point = t_p - len(previous_period) + start_point
        for i in range(start_tau_end_point,len(t)):
            if abs(tau[start_tau_end_point]-tau[i])>stable_threshold/1.5:
                break
        start_tau_end_point = i

        #tau_max
        # max_point = 0
        # for i in np.arange(1, len(following_period)):
        #     if following_period[i] >= following_period[max_point]:
        #         max_point = i
        #     elif (i - max_point) > 2 * expected_frequency:
        #         break
        # t_max=max_point+t_p
        # tau_max = following_period[max_point]

        max_index = [i for i, x in enumerate(period_tau) if x >(max(period_tau)-stable_threshold/1.5)]
        t_max = start + int(np.mean(max_index))
        max_point=t_max-t_p
        tau_max=tau[t_max]


        #tau_1
        end_point = max_point
        for i in np.arange(end_point, len(following_period)):
            if following_period[i] <= following_period[end_point]:
                end_point = i
            elif (i - end_point) > 2 * expected_frequency:
                break
        # end_tau = following_period[end_point]

        end_tau_start_point=t_p+end_point
        for i in range(end_tau_start_point,-1,-1):
            if abs(tau[end_tau_start_point]-tau[i])>stable_threshold/1.5:
                break
        end_tau_start_point=i

        end_tau_end_point = t_p+end_point
        for i in range(end_tau_end_point,len(t)):
            if abs(tau[end_tau_end_point]-tau[i])>stable_threshold/1.5:
                break
        end_tau_end_point=i

        if abs(tau_max-tau_min)<0.01:
            return [0 for i in range(13)]

        # best_fit=1e8
        # best_i=start_tau_end_point
        # for i in range(start_tau_end_point,start_tau_start_point,-1):
        #     slope=(tau[i]-tau_min)/(t_p-i)
        #     fit=np.nanmean([(tau[ii]-(tau[i]-(ii-i)*slope))**2 for ii in range(i,t_p)])
        #     if fit<best_fit:
        #         best_fit=fit
        #         best_i=i
        # start_tau_end_point=best_i
        #
        # best_fit = 1e8
        # best_i = end_tau_start_point
        # for i in range(end_tau_start_point,end_tau_end_point):
        #     slope=(tau_max-tau[i])/(i-max_point)
        #     fit = np.nanmean([(tau[ii] - ((i-ii) * slope + tau[i])) ** 2 for ii in range(max_point, i)])
        #     if fit < best_fit:
        #         best_fit = fit
        #         best_i = i
        # end_tau_start_point = best_i

        start_tau_start_point=start_tau_end_point
        for i in range(start_tau_end_point,-1,-1):
            if abs(tau[start_tau_end_point]-tau[i])<=stable_threshold:
                start_tau_start_point=i
            else:
                break
        end_tau_end_point = end_tau_start_point
        for i in range(end_tau_start_point,len(t)):
            if abs(tau[end_tau_start_point]-tau[i])<=stable_threshold:
                end_tau_end_point=i
            else:
                break

        tau0 = np.nanmean(tau[start_tau_start_point:start_tau_end_point])
        tau1 = np.nanmean(tau[end_tau_start_point:end_tau_end_point])
        slope_0 = - (tau_min - tau[start_tau_end_point]) / (t_p - start_tau_end_point) * expected_frequency * 3600
        slope_1 =  (tau_max - tau_min) / (t_max - t_p) * expected_frequency * 3600
        slope_2 = (tau_max - tau[end_tau_start_point]) / (end_tau_start_point - t_max) * expected_frequency * 3600

        # if min(slope_0,slope_1)<75:
        #     return 0, 0, 0, 0, 0, 0,0,0,0,0

        return [start_tau_start_point,start_tau_end_point,t_p,t_max,end_tau_start_point,end_tau_end_point,
                tau0,tau_min,tau_max,tau1,slope_0,slope_1,slope_2]
    return [0 for i in range(13)]

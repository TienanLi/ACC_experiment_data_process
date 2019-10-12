from side_functions import hex_to_byte,draw_fig,hex_to_int

def analyze_SPEED(message_array):
    speed=[]
    operation_time=[]
    for m in message_array:
        SPEED=int(hex_to_byte(m[1], 71)[47:47 + 16],2)*.01
        speed.append(SPEED)
        operation_time.append(m[0])

    draw_fig(operation_time,'',speed,'speed (kph)')
    return operation_time,speed

def analyze_STEER_ANGLE_SENSOR(message_array):
    steer_angle=[]
    operation_time=[]
    for m in message_array:
        steer=hex_to_int(m[1],71,3,12,signed=True)*1.5
        steer_angle.append(steer)
        operation_time.append(m[0])
    draw_fig(operation_time,'',steer_angle,'steer angle (deg)')

def analyze_LEAD_INFO(message_array):
    lead_rel_speed=[]
    lead_long_dist=[]
    operation_time=[]
    for m in message_array:
        relative_speed=hex_to_int(m[1],71,23,12,signed=True)*.025
        longitudinal_distance=hex_to_int(m[1],71,7,13,signed=False)*.05
        lead_rel_speed.append(relative_speed)
        lead_long_dist.append(longitudinal_distance)
        operation_time.append(m[0])
    draw_fig(operation_time,'',lead_rel_speed,'relative speed (m per second)')
    draw_fig(operation_time,'',lead_long_dist,'space (m)')
    return operation_time,lead_long_dist

def analyze_PCM_CRUISE_2(message_array):
    cruise_active=[]
    set_speed=[]
    operation_time=[]
    for m in message_array:
        active=hex_to_int(m[1],71,15,1,signed=False)
        set_v=hex_to_int(m[1],71,23,8,signed=False)

        cruise_active.append(active)
        set_speed.append(set_v)
        operation_time.append(m[0])
    draw_fig(operation_time,'',cruise_active,'ACC ready to use')
    draw_fig(operation_time,'',set_speed,'set speed (kph)')

    return operation_time,cruise_active

def analyze_PCM_CRUISE(message_array):
    standstill_on=[]
    operation_time=[]
    for m in message_array:
        on=hex_to_int(m[1],71,12,1,signed=False)
        standstill_on.append(on)
        operation_time.append(m[0])
    draw_fig(operation_time,'',standstill_on,'ACC using')

    return operation_time,standstill_on
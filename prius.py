from side_functions import hex_to_byte,draw_fig

def analyze_SPEED(message_array):
    speed=[]
    operation_time=[]
    for m in message_array:
        message = m[1].split('x')[1]

        SPEED=int(hex_to_byte(message, 64)[40:40 + 16],2)*.01


        # length = m[2]
        # SPEED=int(message[10:14],16)*0.01
        speed.append(SPEED)
        operation_time.append(m[0])

    draw_fig(operation_time,'',speed,'speed (kph)')

def analyze_STEER_ANGLE_SENSOR(message_array):
    steer_angle=[]
    operation_time=[]
    for m in message_array:
        steer=int(hex_to_byte(m[1], 72)[39:39 + 4],2)*.1
        steer_angle.append(steer)
        operation_time.append(m[0])
    draw_fig(operation_time,'',speed,'speed (kph)')

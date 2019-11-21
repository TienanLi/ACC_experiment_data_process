from side_functions import hex_to_int,draw_fig

def analyze_DSU_SPEED(message_array):
    forward_speed=[]
    operation_time=[]
    for m in message_array:
        s=hex_to_int(m[1],71,15,16,signed=True)*3.90625
        forward_speed.append(s)
        operation_time.append(m[0])
    draw_fig(operation_time,'',forward_speed,'forward speed (kph)')
    return operation_time,forward_speed
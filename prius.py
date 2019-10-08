from side_functions import hex_to_byte,draw_fig

def analyze_SPEED(message_array):
    speed=[]
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        SPEED=int(message[10:14],16)*0.01
        speed.append(SPEED)

    draw_fig(range(len(speed)),'',speed,'speed (kph)')

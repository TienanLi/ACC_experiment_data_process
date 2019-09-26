from side_functions import hex_to_byte,draw_fig

def analyze_GAS_PEDAL_2(message_array):
    torque_estimate=[]
    torque_request=[]
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        ENGINE_TORQUE_ESTIMATE=int(message[0:4],16)
        ENGINE_TORQUE_REQUEST=int(message[4:8],16)
        torque_estimate.append(ENGINE_TORQUE_ESTIMATE)
        torque_request.append(ENGINE_TORQUE_REQUEST)

    draw_fig(range(len(torque_estimate)),'',torque_estimate,'estimated torque(Nm)')
    draw_fig(range(len(torque_request)),'',torque_request,'request torque(Nm)')

def analyze_ENGINE_DATA(message_array):
    speed=[]
    rpm=[]
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        XMISSION_SPEED=int(message[0:4],16)*0.01
        ENGINE_RPM=int(message[4:8],16)
        speed.append(XMISSION_SPEED)
        rpm.append(ENGINE_RPM)

    draw_fig(range(len(speed)),'',speed,'speed (kph)')
    draw_fig(range(len(rpm)),'',rpm,'RPM')

def analyze_POWERTRAIN_DATA(message_array):
    brake_on = []
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        BRAKE_SWITCH = hex_to_byte(message[4],1)[-1]
        brake_on.append(BRAKE_SWITCH)

    draw_fig(range(len(brake_on)), '', brake_on, 'brake on')

def analyze_STANDSTILL(message_array):
    wheels_moving = []
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        WHEELS_MOVING = hex_to_byte(message[1], 1)[3]
        wheels_moving.append(WHEELS_MOVING)

    draw_fig(range(len(wheels_moving)), '', wheels_moving, 'wheels moving')

def analyze_SEATBELT_STATUS(message_array):
    driver_seatbelt_lamp = []
    driver_seatbelt_unlatched = []
    driver_seatbelt_latched = []
    for m in message_array:
        message = m[1].split('x')[1]
        length = m[2]
        SEATBELT_DRIVER_LAMP = hex_to_byte(message[0], 1)[0]
        SEATBELT_DRIVER_LATCHED = hex_to_byte(message[1], 1)[2]
        SEATBELT_DRIVER_UNLATCHED = hex_to_byte(message[1], 1)[3]
        driver_seatbelt_lamp.append(SEATBELT_DRIVER_LAMP)
        driver_seatbelt_latched.append(SEATBELT_DRIVER_LATCHED)
        driver_seatbelt_unlatched.append(SEATBELT_DRIVER_UNLATCHED)

    draw_fig(range(len(driver_seatbelt_lamp)), '', driver_seatbelt_lamp, 'lamp')
    draw_fig(range(len(driver_seatbelt_latched)), '', driver_seatbelt_latched, 'latched')
    draw_fig(range(len(driver_seatbelt_unlatched)), '', driver_seatbelt_unlatched, 'unlatched')


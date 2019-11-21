import matplotlib.pyplot as plt


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
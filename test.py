from side_functions import hex_to_byte

hex1 = '802408060a000c45'
bin1 = hex_to_byte(hex1,8)
bin2 = bin1[30:40]
print(bin2) # this is to get the 31~40 binary numbers, which correspond to longitudinal acceleration
print(int(bin2,2)) # transform the 31~40 binary number into decimals, and then multiply with factor and add offset
dec1 = int(bin2,2)
factor = -0.035
offset = 17.92
result = factor*dec1 + offset
print('the longitudinal acceleration value is', result)
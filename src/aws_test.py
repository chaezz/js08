#!/usr/bin/env/python3
#
# Analyze and output weather sensor values through Serial communications.
#
# required packages: python3-serial, python3-binascii, python3-struct
#
# TODO(Jongjin): Exception processing necessary.
#

import serial
import time
import binascii
import struct
import sys


def serial_test():
    """The connected weather sensor outputs six values.

    Wind direction, Wind Speed, Temperature, Humidity, AirPressure, Pm2.5
    """
    global wind_speed, i
    try:
        read_aws = b'\x01\x03\x00\x00\x00\x29\x84\x14'
        ser = serial.Serial(port="COM9", baudrate=9600, parity=serial.PARITY_EVEN,
                            stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS,
                            timeout=1)
        if ser.readable():
            ser.write(read_aws)
            res = ser.readline()
            print(f"\nWind direction: {int(binascii.hexlify(res[5:7]), 16)}°")  # Wind direction

            wind_speed_hex = f"{hex(res[8]) + ' ' + hex(res[7]) + ' ' + hex(res[10]) + ' ' + hex(res[9])}"
            temp_hex = f"{hex(res[12]) + ' ' + hex(res[11]) + ' ' + hex(res[14]) + ' ' + hex(res[13])}"
            humid_hex = f"{hex(res[16]) + ' ' + hex(res[15]) + ' ' + hex(res[18]) + ' ' + hex(res[17])}"
            pressure_hex = f"{hex(res[20]) + ' ' + hex(res[19]) + ' ' + hex(res[22]) + ' ' + hex(res[21])}"
            pm25_hex = f"{hex(res[54]) + ' ' + hex(res[53]) + ' ' + hex(res[56]) + ' ' + hex(res[55])}"

            wind_speed_non = wind_speed_hex.replace("0x", '')
            temp_non = temp_hex.replace("0x", '')
            humid_non = humid_hex.replace("0x", '')
            pressure_non = pressure_hex.replace("0x", '')
            pm25_non = pm25_hex.replace("0x", '')

            if wind_speed_non == '0 0 0 0':
                wind_speed = (1,)
                lst_ws = list(wind_speed)
                lst_ws[0] = 0
                wind_speed = tuple(lst_ws)
            else:
                wind_speed = struct.unpack('<f', binascii.unhexlify(wind_speed_non.replace(' ', '')))

            temp = struct.unpack('<f', binascii.unhexlify(temp_non.replace(' ', '')))
            humid = struct.unpack('<f', binascii.unhexlify(humid_non.replace(' ', '')))
            pressure = struct.unpack('<f', binascii.unhexlify(pressure_non.replace(' ', '')))
            pm25 = struct.unpack('<f', binascii.unhexlify(pm25_non.replace(' ', '')))
            print(f"Wind Speed: {round(wind_speed[0], 2)} m/s \nTemperature: {round(temp[0], 2)} °C"
                  f"\nAtmospheric Pressure: {round(pressure[0], 2)} hPa \nHumidity: {round(humid[0], 2)} %"
                  f"\nPM2.5: {round(pm25[0], 2)} ug/m3")

            time.sleep(1)
        ser.close()

    except Exception as e:
        print("Error code: ", e)
        i += 1
        print(f"Error count -> {i}")
        pass


if __name__ == "__main__":
    print("start!")
    i = 0
    while True:
        serial_test()

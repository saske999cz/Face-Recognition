
# This is client code to receive video frames over UDP
import cv2
import imutils
import socket
import numpy as np
import time
import base64
from gpiozero import AngularServo
import smbus

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '169.254.211.205'
print(host_ip)
port = 9999
message = b'Hello Server'
client_socket.sendto(message, (host_ip, port))

# Define some device parameters
I2C_ADDR  = 0x27 # I2C device address
LCD_WIDTH = 16   # Maximum characters per line

# Define some device constants
LCD_CHR = 1 # Mode - Sending data
LCD_CMD = 0 # Mode - Sending command

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x94 # LCD RAM address for the 3rd line
LCD_LINE_4 = 0xD4 # LCD RAM address for the 4th line

LCD_BACKLIGHT  = 0x08  # On
ENABLE = 0b00000100 # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

#Open I2C interface
bus = smbus.SMBus(1)
def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off 
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)
def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = the data
  # mode = 1 for data
  #        0 for command

  bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
  bits_low = mode | ((bits<<4) & 0xF0) | LCD_BACKLIGHT

  # High bits
  bus.write_byte(I2C_ADDR, bits_high)
  lcd_toggle_enable(bits_high)

  # Low bits
  bus.write_byte(I2C_ADDR, bits_low)
  lcd_toggle_enable(bits_low)
def lcd_toggle_enable(bits):
  # Toggle enable
  time.sleep(E_DELAY)
  bus.write_byte(I2C_ADDR, (bits | ENABLE))
  time.sleep(E_PULSE)
  bus.write_byte(I2C_ADDR,(bits & ~ENABLE))
  time.sleep(E_DELAY)
def lcd_string(message,line):
  # Send string to display

  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)
def lcd_servo(name):
#        print(name)
        lcd_init()
        lcd_string("Hello",LCD_LINE_1)
        lcd_string("    "+name ,LCD_LINE_2)
        servo.angle = -90
        time.sleep(4.0)
        lcd_string("Welcome to ",LCD_LINE_1)
        lcd_string("         my home",LCD_LINE_2)
        time.sleep(1.0)
        servo.angle = 90

servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

vid = cv2.VideoCapture(0)
servo.angle = 90
count = 0
while True:
    msg,_ =  client_socket.recvfrom(BUFF_SIZE)
    if msg.decode("utf-8") != 'Hello':
       count = count + 1
       if count == 10:
           name=msg.decode("utf-8")
           lcd_servo(name)
           count = 0
    WIDTH = 400
    _, frame = vid.read()
    frame = imutils.resize(frame, width=WIDTH)
    encoded, buffer = cv2.imencode(
        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    message = base64.b64encode(buffer)
    client_socket.sendto(message, (host_ip, port))
    cv2.imshow('CLIENT', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        client_socket.close()
        break

vid.release()
cv2.destroyAllWindows()




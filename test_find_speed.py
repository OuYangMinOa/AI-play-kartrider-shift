from ctypes                  import wintypes, windll
from ctypes                  import windll
from threading               import Thread
from getKey                  import Key_check
from mss                     import mss
from PIL                     import Image
from pytesseract             import image_to_string

import pytesseract
import win32gui
import warnings
import ctypes
import numpy       as np
import time
import os
import cv2



def callback(hwnd, extra):
    global GAME_TOP
    global GAME_LEFT
    global GAME_WIDTH
    global GAME_HEIGHT


    rect      = win32gui.GetWindowRect(hwnd)
    this_name = win32gui.GetWindowText(hwnd)
    if ("Kart" in this_name):
        # print("Find Kart")
        # print("Window %s:" % this_name)
        hwnd = windll.user32.FindWindowW(0, this_name)
        rect = wintypes.RECT()
        windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
        # print(rect.left, rect.top, rect.right, rect.bottom)

        GAME_TOP    =int(  rect.top  + 6  + ctypes.windll.user32.GetSystemMetrics(4)) 
        GAME_LEFT   =int(  rect.left + 6) 
        GAME_WIDTH  =int(  -(rect.left - rect.right ) - 12)
        GAME_HEIGHT =int(  -(rect.top  - rect.bottom) - 12 -  ctypes.windll.user32.GetSystemMetrics(4))
        # print(GAME_TOP, GAME_LEFT, GAME_WIDTH, GAME_HEIGHT)
def find_screen():
    win32gui.EnumWindows(callback, None)


RECORD_WIDTH = 200
RECORD_HEIGHT = 100

def main():
    global GAME_TOP
    global GAME_LEFT
    global GAME_WIDTH
    global GAME_HEIGHT

    last_time = time.time()
    while True:
        find_screen()

        GAME_TOP    = GAME_TOP + GAME_HEIGHT*0.75
        GAME_LEFT   = GAME_LEFT  + GAME_WIDTH*0.83
        GAME_WIDTH  = GAME_WIDTH  * 0.08
        GAME_HEIGHT = GAME_HEIGHT * 0.1


        mon = {'top': int(GAME_TOP), 'left': int(GAME_LEFT), 'width': int(GAME_WIDTH), 'height': int(GAME_HEIGHT)}
        sct = mss()
        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen  = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen , 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]




        # low_threshold = 100
        # high_threshold = 150
        # masked_edges = cv2.Canny(gray,low_threshold,high_threshold)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        data = image_to_string(thresh, lang='eng',config=' --psm 8 --oem 3 -c tessedit_char_whitelist=0123456789/')
        print(data)

        print("FPS = ",1/abs(last_time-time.time()),end=" ")

        find_screen()
        mon = {'top': int(GAME_TOP), 'left': int(GAME_LEFT), 'width': int(GAME_WIDTH), 'height': int(GAME_HEIGHT)}
        sct = mss()
        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        small_gray = cv2.resize(gray,(RECORD_WIDTH,RECORD_HEIGHT))

        last_time = time.time()
        cv2.imshow('thresh',thresh)
        cv2.imshow('gray',small_gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break



main()
from ctypes                  import wintypes, windll
from ctypes                  import windll
from threading import Thread
from getKey    import Key_check
from mss       import mss
from PIL       import Image
import win32gui
import ctypes
import numpy       as np
import time
import os
import cv2



RECORD_WIDTH  = 480
RECORD_HEIGHT = 270
IS_RGB = True

def callback(hwnd, extra):
    global GAME_TOP
    global GAME_LEFT
    global GAME_WIDTH
    global GAME_HEIGHT


    rect      = win32gui.GetWindowRect(hwnd)
    this_name = win32gui.GetWindowText(hwnd)
    if ("Kart" in this_name):
        print("Find Kart")
        print("Window %s:" % this_name)
        hwnd = windll.user32.FindWindowW(0, this_name)
        rect = wintypes.RECT()
        windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
        # print(rect.left, rect.top, rect.right, rect.bottom)

        GAME_TOP    =int(  rect.top  + 6  + ctypes.windll.user32.GetSystemMetrics(4)) 
        GAME_LEFT   =int(  rect.left + 6) 
        GAME_WIDTH  =int(  -(rect.left - rect.right ) - 12)
        GAME_HEIGHT =int(  -(rect.top  - rect.bottom) - 12 -  ctypes.windll.user32.GetSystemMetrics(4))
        print(GAME_TOP, GAME_LEFT, GAME_WIDTH, GAME_HEIGHT)
def find_screen():
    win32gui.EnumWindows(callback, None)

find_screen()
from ctypes                  import wintypes, windll
from ctypes                  import windll
from threading import Thread
from getKey    import Key_check
from mss       import mss
from PIL       import Image

import win32gui
import warnings
import ctypes
import numpy       as np
import time
import os
import cv2


warnings.filterwarnings('ignore',category=Warning)




# GAME_TOP    = 84
# GAME_LEFT   = 898
# GAME_WIDTH  = 1021
# GAME_HEIGHT = 767

RECORD_WIDTH  = 480
RECORD_HEIGHT = 256
IS_RGB = True

FILEINDEX = 1
FLODERNAME = "dataset"
FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"

os.makedirs(FLODERNAME,exist_ok=True)   

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

def key2onehot(key):
    # "u l r ctrl shift"

    bin_out = [0 for i in range(3)]
    print(key)
    if ("up" in key):
        bin_out[0] = 1
    if ('left' in key):
        bin_out[1] = 1
    if ('right' in key):
        bin_out[2] = 1
    # if ('ctrl' in key):
    #     bin_out[3] = 1
    # if ('shift' in key):
    #     bin_out[4] = 1

    # out = 0
    # for index,value in enumerate(bin_out):
    #     if (value==1):out+= 2**index

    out  = [0 for i in range(6)]
    if (bin_out[0]==1):
        if (bin_out[1]==1):
            out[2] =1
        elif (bin_out[2]==1):
            out[3] =1
        else:
            out[1] =1
    else:
        if (bin_out[1]==1):
            out[4] =1
        elif (bin_out[2]==1):
            out[5] =1
        else:
            out[0] = 1

            

    return out

def save2npy(data):
    print(f"Saving data to {FILENAME}...")
    np.save(FILENAME,data)
    print(f"===== SUCCESSFUL =====")

def main():
    global FILENAME,FILEINDEX
    
    find_screen()

    mon = {'top': GAME_TOP, 'left': GAME_LEFT, 'width': GAME_WIDTH, 'height': GAME_HEIGHT}
    sct = mss()

    while os.path.isfile(FILENAME):   
        FILEINDEX += 1
        FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"

    train_data = []
    print(f"opening new data file {FILENAME}")

    for i in range(5):
        print(f"start in {5-i}s")
        time.sleep(1)

    last_time = time.time()
    while True:

        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))

        if not IS_RGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(RECORD_WIDTH,RECORD_HEIGHT))
        # print("press \"p\" to pause press \"C\" to continue\n press \"S\" to save data\n press \"N\" to create new file")
        key = Key_check()
        frame_choice = key2onehot(key)
        

        if ('p' in key or 'P' in key):
            print("---- pause ---- \n press \"C\" to continue\n press \"S\" to save data\n press \"N\" to create new file")
            print(" press \"D\" to clear saved data")
            while True:
                key = Key_check()
                if ('c' in key or 'C' in key):
                    for i in range(5):
                        print(f"start in {5-i}s")
                        time.sleep(1)
                    break
                if ('s' in key or 'S' in key):
                    FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    Thread(target = save2npy, args = (train_data,), daemon = True).start()

                if ('d' in key or 'D' in key):
                    train_data = []
                    if os.path.isfile(FILENAME):
                        os.remove(FILENAME)
                    print("delete saved data")
                if ('n' in key or 'N' in key):
                    FILEINDEX+= 1
                    FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    while os.path.isfile(FILENAME):   
                        FILEINDEX += 1
                        FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    train_data = []
                    print(f"opening new data file {FILENAME}")
                    time.sleep(1)
                key = []
                time.sleep(0.2)
        print("FPS = ",1/abs(last_time-time.time()),end=" ")
        print(frame_choice)
        last_time = time.time()

        train_data.append([img,frame_choice])
        if (len(train_data) % 250 ==0):
            FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
            Thread(target = save2npy, args = (train_data,), daemon = True).start()

        cv2.imshow('img',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

if __name__=="__main__":
    main()


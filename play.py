
from mss                     import mss
from PIL                     import Image
from matplotlib.pyplot       import *
from MyModel                 import *
from ctypes                  import wintypes, windll
from ctypes                  import windll

import pydirectinput
import pyautogui
import win32gui
import warnings
import ctypes
import torch
import numpy          as np
import time
import cv2
import os

warnings.filterwarnings('ignore',category=UserWarning)

style.use("dark_background")

# os.system('cmd')
# os.system(r'D:\python\pytorch_transformer\Scripts\activate.bat')


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


#########################################################
MODEL_NAME = "models\\wide_resnet101_2_8.pth"
RECORD_WIDTH  = 226
RECORD_HEIGHT = 226

IS_RGB = True
###########################################################

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

        GAME_TOP    =int(   rect.top  + 6  + ctypes.windll.user32.GetSystemMetrics(4)) 
        GAME_LEFT   =int(   rect.left + 6  ) 
        GAME_WIDTH  =int( -(rect.left - rect.right ) - 12)
        GAME_HEIGHT =int( -(rect.top  - rect.bottom) - 12 -  ctypes.windll.user32.GetSystemMetrics(4))
        print(GAME_TOP, GAME_LEFT, GAME_WIDTH, GAME_HEIGHT)

def find_screen():
    win32gui.EnumWindows(callback, None)

def main():

    find_screen()


    mon = {'top': GAME_TOP, 'left': GAME_LEFT, 'width': GAME_WIDTH, 'height': GAME_HEIGHT}
    sct = mss()

    print("Loading model ...")
    # m = model_50() # MyResNet(3,6, array([RECORD_HEIGHT, RECORD_WIDTH])).cuda()
    m = torch.load(MODEL_NAME).to(device)
    # get_3rd_layer_output = K.function([m.layers[0].input],
    #                               [m.layers[5].output])
    print("=== Success ===") 

    
    last_time = time.time()
    last = None

    try:
        while True:
            sct.get_pixels(mon)
            img = np.array( Image.frombytes('RGB', (sct.width, sct.height), sct.image).resize((720,480))  )
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    except KeyboardInterrupt:
        cv2.destroyWindow('img')
        
    while True:

        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))

        #img = grab_screen((GAME_LEFT,GAME_TOP,GAME_WIDTH + GAME_LEFT + 1, GAME_HEIGHT+ GAME_TOP + 1))

        if not IS_RGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(RECORD_WIDTH,RECORD_HEIGHT))

        if not IS_RGB:
            img_pre = torch.from_numpy( img.transpose((2,0,1)).astype(float32)/255 ).to(torch.float32)
        else:
            img_pre = torch.from_numpy(img.transpose((2,0,1)).astype(float32)/255 ).to(torch.float32)


        move = torch.argmax(m(img_pre.unsqueeze(0).cuda() ) ).item() #* np.array([1,0.2,1,0.7,1,1])

        # pydirectinput.keyDown('w',_pause = False)

        ## " none u ul ur r l"
        # print(move)

        if (last != move or move == 0):
            pydirectinput.keyUp('left')
            pydirectinput.keyUp('right')
            pydirectinput.keyUp('up')
            # pydirectinput.keyUp('d')

        if(move==1):
            pydirectinput.keyDown("up",_pause = False)
            print("|")
            # time.sleep(0.3)
            # pydirectinput.keyUp('d')
        if (move==2):
            pydirectinput.keyDown("left",_pause = False)
            pydirectinput.keyDown("up",_pause = False)
            print("<-")

        if (move==3):
            pydirectinput.keyDown("right",_pause = False)
            pydirectinput.keyDown("up",_pause = False)
            print("->")
        if (move==4):
            pydirectinput.keyDown("right",_pause = False)
            print(">")
        if (move==5):
            pydirectinput.keyDown("left",_pause = False)
            print("<")
        # print(key)

        # print(abs(last_time-time.time()))
        # last_time = time.time()

        last = move


        # img_putpur = img_putpur.resize((600, 600))
        # cv2.imshow('img',img)
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
    # show()


if __name__=="__main__":
    main()
    # find_screen()
# 
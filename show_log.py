####
from scipy.signal      import savgol_filter as sgfilter
from numpy             import *
from matplotlib.pyplot import *

import glob
import os

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

all_files = glob.glob('logs/*.npy')
fig, ax = subplots(3)
ax[0].set_title('Train losses')
ax[0].set_ylabel('loss')
ax[0].grid()

ax[1].set_title('Valid losse')
ax[1].set_ylim(0,1)
ax[1].set_ylabel('loss')
ax[1].grid()


ax[2].set_title('Acurracy')
ax[2].set_ylabel('%')
ax[2].set_xlabel('Train step')
ax[2].grid()


for num, each_log in enumerate(all_files):
    MODEl_NAME = os.path.basename(each_log)[:-4]
    print(num+1,MODEl_NAME)
    datas = np.load(each_log,allow_pickle=True)
   

    for index,each_data in enumerate(datas):
        this_x = range(1,len(each_data)+1 )

        if (len(each_data)>51):
        	ax[index].plot(this_x, sgfilter(each_data,51,2) ,"-" , color=colors[num] ,label=MODEl_NAME )
        elif (len(each_data)>3):
        	ax[index].plot(this_x, sgfilter(each_data,3,1) ,"-" , color=colors[num] ,label=MODEl_NAME )
        else:
        	ax[index].plot(this_x, each_data               ,"-" , color=colors[num] ,label=MODEl_NAME )

        ax[index].plot(this_x, each_data                ,"-" ,alpha = 0.2, color=colors[num])
        ax[index].legend(fontsize=8)

fig.autofmt_xdate()
fig.tight_layout()
show()



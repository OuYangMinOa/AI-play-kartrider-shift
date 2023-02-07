####  train with pytorch

from torchvision.transforms import ToTensor, Compose
from matplotlib.pyplot      import *
from torch.utils.data       import random_split
from torch.utils.data       import TensorDataset, DataLoader
from tqdm.auto              import tqdm
from MyModel                import *
from getKey                 import Key_check
from numpy                  import *
from time                   import time, sleep



import torch.optim    as optim
import torchvision
import torch.nn       as nn 
import warnings
import tarfile
import torch
import glob
import os

warnings.filterwarnings('ignore',category=UserWarning)

# os.systm('cmd')
# os.systm(r'D:\python\pytorch_transformer\Scripts\activate.bat')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

def append_flip_data(arr):
    output_img, output_label= [], []
    for img, label in arr:
        img = img
        output_img.append(  (img.transpose((2,0,1)).astype(float32))/255   )
        output_label.append( argmax(label, axis=0))

        if (label[2]==1):
            output_img.append( (flip(img).transpose((2,0,1)).astype(float32))/255)
            output_label.append(3)
            
        if (label[3]==1):
            output_img.append( (flip(img).transpose((2,0,1)).astype(float32))/255)
            output_label.append(2)
        
    return torch.from_numpy(array(output_img)).to(torch.float32), torch.from_numpy(array(output_label)).to(torch.long)


#################################################### Change me #################
data_dir      = 'dataset'
logs_dir      = 'logs'
mode_dir      = 'models'

MODEL_NAME    = "wide_resnet_pre_false"
MODEL_FUNC    =  wide_resnet101_2

RECORD_WIDTH  = 224
RECORD_HEIGHT = 224
batch_size    = 8
num_epoch     = 1000

################################################################################
os.makedirs(data_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)
os.makedirs(mode_dir,exist_ok=True)


dataset_files = glob.glob(data_dir + '/*')


train_losses = []
valid_losses = []
acurracies   = []

model      =  MODEL_FUNC().to(device)
model_size(model)
loss_fn    =  nn.CrossEntropyLoss()
optimizer  =  optim.Adam(model.parameters(),lr=0.001)

print("Press F10 to pause the training")



for epoch in range(1,num_epoch+1):
    
    train_loss = 0.0
    valid_loss = 0.0
    
    
    start_time = time()
    for num,each_file in enumerate(dataset_files[1:]):
        ###################### load data
        try:
            read_file_dataset = load(each_file,allow_pickle=True)
        except:
            continue
        half_len = len(read_file_dataset)//2
        for cut_index in range(2):
            print(f'[*] Epoch:{epoch}  file:{each_file} cut:{cut_index}')
            
            testd_torch_dataset = TensorDataset(
                *append_flip_data(
                    read_file_dataset[(cut_index)*half_len:(cut_index+1)*half_len]
                )
            )
            val_size   = 0.1
            val_size   = int(val_size * len(testd_torch_dataset))
            train_size = len(testd_torch_dataset) - val_size
            train_ds, val_ds = random_split(testd_torch_dataset,[train_size, val_size])

            del testd_torch_dataset
            train_dl = DataLoader(train_ds, batch_size,shuffle=True, num_workers=0,pin_memory=True)
            del train_ds
            val_dl   = DataLoader(val_ds  , batch_size,shuffle=True, num_workers=0,pin_memory=True)
            del val_ds



            ###################### train
            model.train()
            for img, lbl in tqdm(train_dl):
                img = img.to(device)
                lbl = lbl.to(device)
                optimizer.zero_grad()
                predict = model(img)
                loss = loss_fn(predict,lbl)
                loss.backward()
                optimizer.step()
                train_loss  +=  loss.item()*img.size(0)


                ###################### pause training ########################
                key = Key_check()  
                if ('y' in key):  ## press f10 to pause
                    print("-------- pause -------- \n press \"C\" to continue")
                    while True:
                        key = Key_check()
                        if ('c' in key or 'C' in key):
                            break
                        sleep(0.2)
                #############################################################   

            ###################### val test
            model.eval()
            accuracy = 0.0
            total = 0.0

            for img, lbl in val_dl:
                img = img.to(device)
                lbl = lbl.to(device)

                predict = model(img)
                loss = loss_fn(predict,lbl)

                valid_loss += loss.item()*img.size(0)


                # calculate accuracy
                _, predicted = torch.max(predict.data, 1)
                total       += lbl.size(0)
                accuracy    += (predicted == lbl).sum().item()

            accuracy = (100 * accuracy / total)


            train_loss = train_loss/len(train_dl.sampler) 
            valid_loss = valid_loss/len(val_dl.sampler)

            del train_dl
            del val_dl

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            acurracies.append(accuracy)
            
            torch.save(model, f"{mode_dir}/{MODEL_NAME}_{epoch}.pth")
            np.save(f"{logs_dir}/{MODEL_NAME}_losses.npy",[train_losses,valid_losses,acurracies])
            print('\t\t Loss:{:.4f}, valid Losss:{:.4f}, val acc:{:.2f}%\n\n'.format(
                train_loss,valid_loss,accuracy
            ))  
    print("="*30,f"Epoch:{epoch} Finish",'='*30)
    print(f'\t Spend {time() - start_time}s\n')
# print(f"Spend {total_time}s")




























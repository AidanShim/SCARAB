import numpy as np
from microboone_utils import *
from file import File
from skimage.measure import block_reduce
from math import floor, ceil
import gc

f_downsample = 10
h_split = 40
v_split = 192
nbatch = 64

adccutoff = 10.*f_downsample/10.
adcsaturation = 100.*f_downsample/10.
tables = ['wire_table']

for filenum in range(1):  # 0 to 17
    file_path = f"./inputData/bnb_WithWire_{filenum:02d}.h5"
    f = File(file_path)
    f_size = len(f)
    print(f'File:{filenum}, nBatch:{f_size/nbatch}')
    for t in tables: f.add_group(t)

    for startEvt in range(0, f_size, nbatch):
        images = np.zeros((nbatch*h_split*v_split,nwires(2)//v_split,ntimeticks()//f_downsample//h_split,1))

        f.read_data(startEvt, nbatch)
        evts = f.build_evt()

        for idx in range(len(evts)):
            print(f"Event: {idx}")
            evt = evts[idx]
            wires = evt.__getitem__('wire_table')
            planeadcs = [wires.query("local_plane==%i"%p)[['adc_%i'%i for i in range(0,ntimeticks())]].to_numpy() for p in range(2,nplanes())]

            planeadcs = block_reduce(planeadcs[0], block_size=(1,f_downsample), func=np.sum)
            
            adccutoff = 10.*f_downsample/6.
            adcsaturation = 100.*f_downsample/6.
            planeadcs[planeadcs<adccutoff] = 0
            planeadcs[planeadcs>adcsaturation] = adcsaturation


            X = np.array(np.split(np.array(planeadcs), h_split, axis=1))
            X = np.array(np.split(np.array(X), v_split, axis=1))
            X = np.reshape(X, (-1,nwires(2)//v_split,ntimeticks()//h_split//f_downsample,1))

            X_shape = X.shape    
            images[idx * X_shape[0]: (idx + 1) * X_shape[0]] = X
            del planeadcs, X
            gc.collect() #garbage collector collect

        print("saving")
        np.save(f'./inputData/bnb_WithWire_{filenum:02d}_batch_{startEvt//nbatch:02d}_{h_split}X{v_split}.npy', images)
        del images
        gc.collect() #garbage collector collect
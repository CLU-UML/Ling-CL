import os, sys
import numpy as np

name = sys.argv[1]

files = [x for x in os.listdir('losses') if x.startswith(name + '_')]

losses = [np.load('losses/' + fn) for fn in files]

train = np.mean([loss['train'] for loss in losses], 0)
dev = np.mean([loss['dev'] for loss in losses], 0)
test = np.mean([loss['test'] for loss in losses], 0)

np.savez('losses/' + name, train = train, dev = dev, test = test)

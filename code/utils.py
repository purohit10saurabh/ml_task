from PIL import Image
import pdb
import numpy as np
import os
import sys 
import torch
import random as rd
import time

file_path = os.path.realpath(__file__)

workspace = os.path.abspath(file_path + "/../../../")

data_dir = workspace + '/data/'

images_path = data_dir + 'recorded_images/'

print("Make sure images are in",images_path,"and actions are in",data_dir + 'actions.npy')

def read_data(path = images_path):
	ls = []
	for file in os.listdir(path):
		ls.append(int(file.strip(".png")))

	dps = len(ls)
	ls.sort()
	return ls

def read_split(filename = 'split.txt'):
	with open(data_dir + filename) as f:
		arr = np.loadtxt(f)
	return arr.astype(int)

def create_split(path = images_path, file = data_dir + 'split.txt', split_ratio=0.1):
	# in split file, 0 is training and 1 is testing	
	ls = read_data()
	dps = len(ls)

	rd.seed(0)
	rd.shuffle(ls)

	split = np.zeros(dps)

	split[ls[:int(split_ratio*dps)]] = 1

	with open(file,"w") as f:
		np.savetxt(f, split, fmt ='%.0f')

def get_normalize_stats(images_path, trn_dps):
	it = 0
	arr = np.zeros(0)
	for el in trn_dps:
		vec = np.asarray(Image.open(images_path + str(el) + '.png'))
		assert vec.shape == (84,84,4)
		arr = np.append(arr,vec.flatten())
		it+=1
		if it > 1e3:
			break
	return np.mean(arr), np.std(stdev)

def write_stats(path=data_dir + 'stats.txt'):
	actions = np.load(data_dir + 'actions.npy')
	ls = read_data()
	dps = len(ls)
	with open(path, "w") as f:
		f.write("Number of datapoints = " + str(dps) + "\n")
		for i in range(np.max(actions)+1):
			f.write( "Number of datapoints with action " + str(i) + " = " + str(np.sum(actions==i)) + "\n")
	print("Stats are at", path)

class batch_loader():
	def __init__(self, path = images_path, batch_size = 32):
		self.batch_size = batch_size
		self.path = path
		self.split = read_split()
		trn_dps = np.sort(np.where(self.split == 0)[0])
		rd.seed(0)
		rd.shuffle(trn_dps)
		self.trn_dps = trn_dps
		self.batches = [trn_dps[i:i+batch_size] for i in range(0, len(trn_dps), batch_size)]
		self.num_batches = len(self.batches)
		print("Batch size is",batch_size,"num batches is",self.num_batches)
		self.mean, self.stdev = 32.3, 56.2 #get_normalize_stats(images_path, trn_dps)

	def get_data(self, batch_num):
		if batch_num=='test':
			batch = np.where(self.split == 1)[0]
		elif batch_num=='trn':
			batch = self.trn_dps
		else:
			batch = self.batches[batch_num]

		data = []
		for el in batch:
			arr = np.asarray(Image.open(images_path + str(el) + '.png'))
			arr = (arr-self.mean)/self.stdev
			data.append(arr)
		
		data = np.array(data)
		#currently in b h w c format
		data = np.moveaxis(data, (3, 1), (1,3)) #pytorch convention b c w h
		return torch.from_numpy(data).float()	


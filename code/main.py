import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = 'cuda:0,1' if torch.cuda.is_available() else 'cpu'
print("Must have atleast 2 gpus for training, Device is",device)

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(4, 6, 5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 16, 5)
		self.fc1   = nn.Linear(784, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84,4)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		return np.prod(size)

def backprop_deep(Data, labels, net, T=50, lr=0.001, weight_decay=0.0):
	print(T,"epochs on",device)
	'''
	Backprop
	Args:
		D: Data
		net: neural network
		T: number of epochs
		lr: learning rate for Adam
		weight_decay: weight decay
	'''
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
	print("Using",optimizer)

	print("Loading training data...")
	trn_data = Data.get_data('trn')
	print("Training started.")
	
	for epoch in range(T):
		print("Epoch",epoch+1)
		running_loss = 0
		trn_acc = 0
		for batch_num in range(Data.num_batches):
			inputs = trn_data[batch_num*Data.batch_size : (batch_num+1)*Data.batch_size].to(device)
			labs = labels[torch.from_numpy(Data.batches[batch_num])]
			
			# Initialize the gradients to zero
			optimizer.zero_grad()

			# Forward propagation
			outputs = net(inputs)

			# Error evaluation
			loss = criterion(outputs, labs)

			# Back propagation
			loss.backward()

			# Parameter update
			optimizer.step()

		#print('[epoch %d] trn acc: %.3f loss: %.3f' %(epoch+1, trn_acc/Data.num_batches, running_loss / Data.num_batches)) 

def eval(Data, model_path, ltest, device=device):
	print('eval using',device)
	net = torch.load(model_path)
	net.eval() 
	with torch.no_grad():
		lpred = net(Data.get_data('test').to(device))
	acc = float(100 * (ltest==lpred.max(1)[1]).float().mean())
	return acc

if __name__=="__main__":
	write_stats()  #writing dataset stats
	create_split() # create trn tst split
	Data = batch_loader(batch_size=16)
	labels = torch.from_numpy(np.asarray(np.load(data_dir + 'actions.npy').flatten())).to(device)
	ltest = labels[torch.from_numpy(np.where(Data.split == 1)[0])]
	model_path = os.path.realpath(file_path + "../../../model.pth")
	print("model_path is",model_path) 

	mode = 'train'
	if mode=='train':
		net = LeNet()
		net = nn.DataParallel(net, device_ids=[0,1])
		net.to(device)
		print(net)
		print("Total parameters are", sum(p.numel() for p in net.parameters() if p.requires_grad))

		net.eval() 
		with torch.no_grad():
			lpred = net(Data.get_data('test').to(device))
		print("Random model Test Accuracy is",100 * (ltest==lpred.max(1)[1]).float().mean())

		start = time.time()
		net.train()
		backprop_deep(Data, labels, net, T=10, lr=0.0001)
		end = time.time()
		print(f'It takes {end-start:.6f} seconds.')
		torch.save(net, model_path)

	mode = 'eval'
	if mode=='eval':
		print("Test Accuracy is",eval(Data, model_path, ltest))
	

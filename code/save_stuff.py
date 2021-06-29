from main import *

device = 'cpu'

def convert_cpu(path, out_path='../cpu_model.pt'):
	model = torch.load(path).module
	#pdb.set_trace()
	model = model.to(device)
	torch.save(model,out_path)

Data = batch_loader()
labels = np.asarray(np.load(data_dir + 'actions.npy').flatten())
ind = np.where(Data.split == 1)[0].astype(np.int)
rd.seed(0)
rd.shuffle(ind)
ltest = labels[ind].astype(np.int)
np.savetxt(data_dir + 'test_dps.txt', ind, fmt='%i')
np.savetxt(data_dir + 'test_actions.txt', ltest, fmt='%i')
#acc = eval(Data, path, ltest, device=device)

model_path = os.path.realpath(os.path.dirname(__file__) + "../model.pth")
#pdb.set_trace()
convert_cpu(model_path)


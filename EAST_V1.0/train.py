import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from test_dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np
import visdom

class Visualizer:

    def __init__(self, env = 'EAST', **kwargs):

        self.vis = visdom.Visdom(env = env, use_incoming_socket= False, **kwargs)
        self.index = {}


    def plot(self,  y):

        x = np.arange(len(y))
        self.vis.line(Y = np.array(y),X =x,
                      win = 'loss',
                      opts=dict(title='loss')
                     )

visual = Visualizer()

def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval,model_path):

	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	if model_path is not None:
		model.load_state_dict(torch.load(model_path))
		print('----------model load success-------')
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)
	collection_loss = []

	for epoch in range(epoch_iter):	
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		collection_loss.append( epoch_loss/int(file_num/batch_size) )
		visual.plot(collection_loss)
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
	# train_img_path = os.path.abspath('./dataset/img')
	# train_gt_path  = os.path.abspath('./dataset/gt')
	train_img_path = r"F:\training_data\img"
	train_gt_path = r"F:\training_data\gt"

	# model_path = './pths/model_epoch_365.pth'
	model_path = None
	pths_path      = './pths'
	batch_size     = 32
	lr             = 1e-3
	num_workers    = 16
	epoch_iter     = 1000
	save_interval  = 50
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, model_path)
	

import os
import torch
import torchvision.transforms as transforms
import cv2
import sys
import numpy as np
import datetime
import random
import math
import  time
import torch.optim as optim
import torch.nn as nn
from random import sample, shuffle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F 
import torch.distributed as dist
import matplotlib.pyplot as plt

from PIL import Image

from .loss import get_loss, get_loss2, get_loss_multi, get_loss_test, get_loss_of, get_loss_multi_multi, get_loss_multidd
from .draw import disp2rgb, flow_uv_to_colors, flow_to_image, viz_feature

from dataloader.load import load_calib_cam_to_cam, readFlowKITTI, disparity_loader, triangulation

import dataloader.dataset as datasets

class TTCTrainer(object):
    def __init__(self, model, dataset, optimizer, args, device, model_path = None, save_path=None, 
                start_epoch=0, test_model=None, parallel=False, time_stamp=None, max_lr=1e-4, crop_size=[352,1152]):
        self.model = model
        self.test_model = test_model
        self.parallel = parallel
        self.batch_size = args.batch_size
        self.parallel = parallel
        self.train_sampler = None
        if not self.parallel:
            self.train_loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=True, 
                            num_workers=args.batch_size, drop_last=True, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(dataset)
            self.train_loader = DataLoader(dataset,batch_size=args.batch_size, \
                                sampler=self.train_sampler, shuffle=False, pin_memory=True, num_workers=2)

        self.epoch = args.epoch
        # if optimizer is None:
        #     self.optimizer = self.get_optimizer()
        # else :
        self.optimizer = optimizer
        self.start_epoch = start_epoch

        self.save_path = save_path
        
        steps_per_epoch = int(len(self.train_loader))
        self.steps_per_epoch = steps_per_epoch
        # print('xxxxx', self.epoch, len(self.train_loader), self.batch_size, steps_per_epoch)
        starte = -1
        if self.start_epoch>0:
            starte = self.start_epoch
        self.lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr,
            epochs=self.epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=max(int(steps_per_epoch*starte),-1),
        )

        self.device = device
        self.train_loss_history = []
        self.plt_train_epoch = []
        if time_stamp is None:
            self.time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
            out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
        else:
            self.time_stamp = time_stamp
        self.model_path = model_path
        
        self.attn_type=args.attn_type
        self.attn_splits_list=args.attn_splits_list
        self.corr_radius_list=args.corr_radius_list
        self.prop_radius_list=args.prop_radius_list
        self.num_reg_refine=args.num_reg_refine

        self.grad_clip = 1.0
        self.checkpoint = 1

        self.loss_per_epoch = 0
        self.loss_sum_per_epoch = 0
        self.iters = 0

    def get_optimizer(self):
        params = list(self.model.named_parameters())
        param_group = [
            {'params':[p for n,p in params if 'featnet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'flownet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'scalenet' in n],'lr':1e-4},
        ]
        optimizer = torch.optim.Adam(param_group,lr=1e-5)
        return optimizer

    def train(self):
        start = time.time()
        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
        
        # file = open(loss_txt,'w')
        # file.close()
        #self.lr_scheduler2.step()
        # print("Learning rate: ", self.optimizer.state_dict()['param_groups'][0]['lr'])
        # loss_now, f1, epe, d1 = self.eval_epoch()
        # print('Eval: ', loss_now, ' ', f1, ' ', epe, ' ', d1)
        for epoch in range(self.start_epoch, self.epoch):
            print('Epoch:', epoch)
            self.loss_per_epoch = 0
            self.loss_sum_per_epoch = 0
            self.f1 = 0
            self.d1 = 0
            self.iters = 0
            # loss_now
            # if torch.distributed.get_rank()==0:
            #print(('Train %f %f '%(self.loss_per_epoch/max(1,self.iters), self.loss_sum_per_epoch/max(1,self.iters))))
            self.train_epoch(epoch)
            loss_now, f1, epe, d1 = self.eval_epoch()
            print('Eval: ', loss_now, ' ', f1, ' ', epe, ' ', d1)
            if (self.parallel and torch.distributed.get_rank()==0) or (not self.parallel):
                loss_txt = out_dir + '/0.txt'
                file = open(loss_txt,'a')
                file.write('Loss in epoch %d: '%(epoch))
                file.write('Train %f %f %f %f '%(self.loss_sum_per_epoch/max(1,self.iters),\
                                             self.loss_per_epoch/max(1,self.iters), \
                                            self.f1/max(1,self.iters), self.d1/max(1,self.iters))) 
                # file.write('Train %f' %(self.loss_per_epoch/max(1,self.iters)))   
                file.write('Test %f %f %f %f\n'%(loss_now, f1, epe, d1))          
                file.close()

                if (epoch<self.epoch and epoch%self.checkpoint==0):
                    checkpoint = {
                            "net": self.model.state_dict(),
                            'optimizer':self.optimizer.state_dict(),
                            "epoch": epoch+1
                        }
                    temp_pth = self.model_path[:-8] + str(epoch) + 'y.pth.tar'
                    # temp_pth = self.model_path[:-10] + '_y' + str(epoch) + '.pth.tar'
                    torch.save(checkpoint, temp_pth)
                
                print("Loss in epoch", epoch, ":", self.loss_per_epoch/max(1,self.iters))
                print("Learning rate: ", self.optimizer.state_dict()['param_groups'][0]['lr'])

        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            "epoch": 0
        }
        temp_pth = self.save_path
        torch.save(checkpoint, temp_pth)

        # end = time.time()
        # total_time = end-start
        # print(f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s')
        # torch.save(self.model.state_dict(), self.model_path)
    
    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        ten_loss_sum = 0

        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
        save_index = 2000 if not self.parallel else random.randint(int(1000/self.batch_size),int(2000/self.batch_size))
        t0 = time.time()
        for i, data in enumerate(self.train_loader):
            # if i<=4780 or i>=4800:
            #     print(i)
            #     continue
            #print('aaaaaaaaaaaaaaaaaa', time.time()-t0)
            #print('xxxxxxxxxxxxxxxxxx', i)
            img0, img1, flow_gt, imgAux, disp_gt, disp_gt2, valid, ts = data
            img0, img1 = img0.to(self.device), img1.to(self.device)
            # flow_gt ,imgAux = flow_gt.to(self.device), imgAux.to(self.device)
            imgAux = imgAux.to(self.device)
            disp_gt = disp_gt.to(self.device)
            disp_gt2 = disp_gt2.to(self.device)
            # print(torch.max(disp_gt), torch.min(disp_gt), torch.mean(disp_gt))
            valid = valid.to(self.device)
            flow_gt = flow_gt.to(self.device)
            ts = ts.to(self.device)
            start_time = time.time()
            #print(img0.shape)
            self.optimizer.zero_grad()
            # for param in self.model.parameters():
            #     param.grad = None
            
            time2 = time.time()

            scale, disp, disp2, flow, flow0, ffeat, sfeat = self.model(img0, img1,
                                    attn_type=self.attn_type,
                                    attn_splits_list=self.attn_splits_list,
                                    corr_radius_list=self.corr_radius_list,
                                    prop_radius_list=self.prop_radius_list,
                                    num_reg_refine=self.num_reg_refine,
                                    )

            end_time = time.time()
            #print(end_time-time2)
            # print(scale.shape, flow.shape)
            # scale_gt_selfsup = self_supervised_gt_affine(flow)[0]
            # scale_gt_selfsup[~(valid[0].unsqueeze(0).bool())] = 0
            if type(flow) == list:
                if type(scale) == list:
                    loss, loss_last, loss_d, gt_scale, f1, d1 = get_loss_multidd(scale, disp, disp2, flow, imgAux, disp_gt, disp_gt2, flow_gt, valid, epoch)
                else:
                    loss, loss_last, gt_scale, f1 = get_loss_multi(scale, flow, imgAux, flow_gt, valid, epoch)
            elif flow0 is not None:
                loss, loss_last, gt_scale, f1 = get_loss2(scale, flow, flow0, imgAux, flow_gt, valid, epoch)
            else:
                # loss, loss_last, gt_scale, f1 = get_loss_of(scale, flow, imgAux, flow_gt, valid, ts)
                loss, loss_last, gt_scale, f1 = get_loss(scale, flow, imgAux, flow_gt, valid, epoch)
            # if ts>0:
            #     if type(scale) == list:
            #         loss, loss_last, gt_scale, mask = get_loss_multi(scale, imgAux, epoch)
            #     elif flow0 is not None:
            #         loss, loss_last, gt_scale, f1 = get_loss2(scale, flow, flow0, imgAux, flow_gt, valid, epoch)
            #     else:
            #         loss, loss_last, gt_scale, f1 = get_loss(scale, flow, imgAux, flow_gt, valid, epoch)
            # else:
            #         loss, loss_last, gt_scale, f1 = get_loss_of(scale, flow, imgAux, flow_gt, valid, epoch)

            # torch.distributed.barrier()
            time3 = time.time()
            # loss += ttc_smooth_loss(img1, scale, torch.ones_like(valid))
            loss_last.backward()
            if torch.isnan(loss_last):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # torch.distributed.barrier()
            self.lr_scheduler3.step()
            #print(time.time()-time3, time3-time2, time2-time1)

            #print(flow.shape, scale.shape, gt_scale.shape, img0.shape)
            if type(scale) == list:
                scale = scale[-1]
            if type(flow) == list:
                flow = flow[-1]
            if type(disp) == list:
                disp = disp[-1]
            if i%int(save_index)==0:
                if not self.parallel or (self.parallel and torch.distributed.get_rank()==0):
                    disp_gt_show = (disp_gt[0,...]).transpose(0,1).transpose(1,2).squeeze().detach().cpu().numpy()
                    disp_show = (disp[0,...]).transpose(0,1).transpose(1,2).squeeze().detach().cpu().numpy()
                    disp_result = cv2.vconcat([disp_show, disp_gt_show])
                    plt.imsave(os.path.join(out_dir, str(epoch)+'_'+str(i)+'dis_cmp'+'.jpg'), disp_result, cmap='magma')

                    out_of_viz = flow_to_image((flow.permute(0,2,3,1).detach().cpu().numpy())[0])
                    out_of_viz_gt = flow_to_image((flow_gt.permute(0,2,3,1).detach().cpu().numpy())[0])
                    of_result = cv2.vconcat([out_of_viz, out_of_viz_gt])
                    cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'flow_cmp'+'.jpg'), of_result)

                    ttc_warp_image2 = ((scale[0]).transpose(0,1).transpose(1,2) - 0.5) / (1.0)
                    ttc_warp_image2 = disp2rgb(np.clip(ttc_warp_image2.detach().cpu().numpy(), 0.0, 1.0))
                    ttc_warp_image2 = ttc_warp_image2*255.0
                    ttc_warp_image = ((gt_scale[0,...]).transpose(0,1).transpose(1,2) - 0.5) / (1.0)
                    ttc_warp_image = disp2rgb(np.clip(ttc_warp_image.detach().cpu().numpy(), 0.0, 1.0))
                    ttc_warp_image = ttc_warp_image*255.0
                    oe_result = cv2.vconcat([ttc_warp_image2, ttc_warp_image])
                    cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'scale_cmp'+'.jpg'), oe_result)


                    img = ((img0[0]).transpose(0,1).transpose(1,2))
                    img = np.clip(img.detach().cpu().numpy(), 0.0, 255.0)
                    img1 = ((img1[0]).transpose(0,1).transpose(1,2))
                    img1 = np.clip(img1.detach().cpu().numpy(), 0.0, 255.0)
                    img = cv2.vconcat([img, img1])
                    cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'imgs'+'.jpg'), img)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    ffeat_viz = viz_feature(ffeat)
                    plt.imsave(os.path.join(out_dir, str(epoch)+'_'+str(i)+'f'+'.jpg'), ffeat_viz)
                    sfeat_viz = viz_feature(sfeat)
                    plt.imsave(os.path.join(out_dir, str(epoch)+'_'+str(i)+'s'+'.jpg'), sfeat_viz)

            # if i%(int(self.steps_per_epoch/4))==0 and i>0 and i<(self.steps_per_epoch-20):

            #     out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
            #     loss_txt = out_dir + '/1.txt'
            #     loss_now, f1x, epe, d1x = self.eval_epoch()
            #     print('Eval: ', loss_now, ' ', f1x, ' ', epe, ' ', d1x)
                
            #     if (self.parallel and torch.distributed.get_rank()==0) or (not self.parallel):
            #         file = open(loss_txt,'a')
            #         file.write('Loss in epoch %d %d: '%(epoch, self.iters))
            #         file.write('Train %f %f %f %f '%(self.loss_sum_per_epoch/max(1,self.iters),\
            #                                     self.loss_per_epoch/max(1,self.iters), \
            #                                     self.f1/max(1,self.iters), self.d1/max(1,self.iters))) 
            #         # file.write('Train %f' %(self.loss_per_epoch/max(1,self.iters)))   
            #         file.write('Test %f %f %f %f'%(loss_now, f1x, epe, d1x))    
            #         file.write('Lr %f\n'%((self.optimizer.state_dict()['param_groups'][0]['lr'])*1e4))      
            #         file.close()
            #         # if loss_now < 0.004460:
            #         if (epoch<self.epoch):
            #             checkpoint = {
            #                     "net": self.model.state_dict(),
            #                     'optimizer':self.optimizer.state_dict(),
            #                     "epoch": epoch+1
            #                 }
            #             temp_pth = self.model_path[:-8] + str(epoch) + '_' + str(self.iters) + 'y.pth.tar'
            #             # temp_pth = self.model_path[:-10] + '_y' + str(epoch) + '.pth.tar'
            #             torch.save(checkpoint, temp_pth)
                        
            # if (self.parallel and torch.distributed.get_rank()==0):
            #     dist.all_reduce(loss, op = dist.ReduceOp.SUM)
            #     loss /= float(dist.get_world_size())
            #     if i % 10 == 0:
            #         print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
            #             ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
            #             '{:6.4f}'.format(loss.item()))
            # else:
            if loss_last is not None:
                self.loss_per_epoch += loss.item()
                self.loss_sum_per_epoch += loss_last.item()
                self.f1 += f1.item()
                self.d1 += d1.item()
                if i % 10 == 0:
                        print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
                            ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
                            '{:6.4f}'.format(loss_last.item()) + '    ' + '{:6.4f}'.format(loss.item()), '    ' + '{:6.4f}'.format(f1.item()) + '    ' +  '{:6.4f}'.format(loss_d.item()))
            else:
                self.loss_per_epoch += loss.item()
                if i % 100 == 0:
                    print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
                        ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
                        '{:6.4f}'.format(loss.item()))

                                   
            # if i % 10 == 0:
            #     print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
            #         ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
            #         '{:6.4f}'.format(loss.item()) + '   Loss eva: ' + '{:6.4f}'.format(loss2.item()))

            # self.loss_per_epoch += loss2.item()
            self.iters += 1

            t0 = time.time()
                
            #print('Epoch Time: ', end_time-start_time)
        
    @torch.no_grad()
    def eval_epoch(self, eval_path='/mnt/pool/Datasets/OpticalFlow/kitti/data_scene_flow/training/'):

        img0x, img1x, flow0 = self.eva_dataloader(eval_path)
        # print('xxxxx', len(img0x))
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flow0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flow0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flow0]

        w0,h0 = 1216,352
        # w0,h0 = 960,288
    #     # w0,h0 = 768, 352

        total_loss=0
        f1_mean = 0
        d1_mean = 0
        epe_mean = 0
        f1 = 0
        d1 = 0
        t=0

        self.model.eval()

        for i in range(len(img0x)):
            # print(flow0[i])
            flow_gt, valid_ori = readFlowKITTI(flow0[i])
            ints = load_calib_cam_to_cam(calib[i])
            fl = ints['K_cam2'][0,0]
            cx = ints['K_cam2'][0,2]
            cy = ints['K_cam2'][1,2]
            bl = ints['b20']-ints['b30']
            d1 = disparity_loader(disp0[i])
            d2 = disparity_loader(disp1[i])

            flow_gt = np.ascontiguousarray(flow_gt,dtype=np.float32)
            flow_gt[np.isnan(flow_gt)] = 1e6 # set to max
            valid = np.logical_and(np.logical_and(valid_ori>0.99, d1>0), d2>0)
            d1[d1<=0] = 1e6
            d2[d2<=0] = 1e6

            shape = d1.shape
            mesh = np.meshgrid(range(shape[1]),range(shape[0]))
            xcoord = mesh[0].astype(float)
            ycoord = mesh[1].astype(float)
            
            # triangulation in two frames
            P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl = fl, cx = cx, cy = cy)
            P1 = triangulation(d2, xcoord + flow_gt[:,:,0], ycoord + flow_gt[:,:,1], bl=bl, fl = fl, cx = cx, cy = cy)
            dis0 = P0[2]
            dis1 = P1[2]

            change_size =  dis0.reshape(shape).astype(np.float32)
            valid = np.logical_and(valid, change_size>0).astype(float)
            flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))
            change_size = np.concatenate((change_size[:,:,np.newaxis],flow3d),2)
            scale_gt = (np.array(change_size).astype(np.float32))

            gt_depth = scale_gt[...,0:1]
            # gt_depth[gt_depth<=0] = 1e6
            gt_f3d =  scale_gt[...,1:]
            gt_dchange = (1+gt_f3d[...,2:]/gt_depth)
            maskdc = (gt_dchange < 3) & (gt_dchange > 0.3) & np.expand_dims(valid.astype(bool),axis=2)


            file_1 = img0x[i]
            file_2 = img1x[i]

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)    
            ori_size = image1.shape[0:2]      
            # print(ori_size)
            # image1 = cv2.resize(image1, (w0,h0))
            # image2 = cv2.resize(image2, (w0,h0))
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)


            padding_factor = 32
            inference_size = [h0,w0]

            
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                # print(inference_size, ori_size)
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                        align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                        align_corners=True)
            scale, disp, disp2, flow, _ = self.model(image1, image2,
                                    attn_type=self.attn_type,
                                    attn_splits_list=self.attn_splits_list,
                                    corr_radius_list=self.corr_radius_list,
                                    prop_radius_list=self.prop_radius_list,
                                    num_reg_refine=self.num_reg_refine,
                                    testing=True,
                                    )
            if type(flow) == list:
                flow = flow[-1]
            if type(scale) == list:
                scale = scale[-1]
            if type(disp) == list:
                disp = disp[-1]
                # print('xxx')
            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                scale = F.interpolate(scale, size=ori_size, mode='bilinear',
                                        align_corners=True)
                disp = F.interpolate(disp, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow = F.interpolate(flow, size=ori_size, mode='bilinear',
                                        align_corners=True)


                flow[:,0:1,...] = flow[:,0:1,...] * ori_size[-1] / inference_size[-1]
                flow[:,1:,...] = flow[:,1:,...] * ori_size[0] / inference_size[0]
            
            scale = scale[0].transpose(0,1).transpose(1,2).detach().cpu().numpy()
            h,w,_ = gt_dchange.shape
            # if scale.shape[0]!=h and scale.shape[1]!=w:
            #     scale = cv2.resize(scale, (w,h))
            # scale = np.expand_dims(scale, axis=2)
            loss =  (np.abs((np.log(scale)-np.log((gt_dchange)))))[maskdc].mean()
            total_loss += loss.mean()

            #flow_loss
            flow_gt = torch.as_tensor(flow_gt).to('cpu').permute(2,0,1)

            flow = flow[0].to('cpu')
            disp = disp[0].to('cpu')
            # print(flow.shape, disp.shape, scale.shape)

            valid_ori = torch.as_tensor(valid_ori).to('cpu')
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            # print(mag.shape, valid_ori.shape)
            valid = (valid_ori >= 0.5) & (mag < 400)

            mean_disp = np.mean(d1[valid==1])
            d1[valid==1] = d1[valid==1] / mean_disp
            disp_gt = torch.as_tensor(d1).to('cpu')
            disp_gt = np.expand_dims(disp_gt, axis=0)

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid.view(-1) >= 0.5
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            f1 = torch.mean(out[val])
            epe = torch.mean(epe[val])

            eped = torch.sum(torch.abs(disp - disp_gt), dim=0)
            # print(eped.shape)
            eped = eped.view(-1)
            # magd = torch.from_numpy(disp_gt).abs()
            # magd = magd.view(-1)
            # outd = ((eped > 3.0) & ((eped/magd) > 0.05)).float()
            # print(eped.shape, val.shape)
            d1 = torch.mean(eped[val])

            f1_mean += f1.item()
            d1_mean += d1.item()
            epe_mean += epe.item()
            t += 1

        loss = total_loss / float(t)
        f1 = f1_mean / float(t)
        d1 = d1_mean / float(t)
        epe = epe_mean / float(t)

        self.model.train()

        return loss, f1, epe, d1

    # @torch.no_grad()
    # def eval_epoch(self):

    #     test_dataset = datasets.KITTI_160()
    #     w0,h0 = 1216,352
    #     # w0,h0 = 960,288
    #     # w0,h0 = 768, 352

    #     total_loss=0
    #     f1_mean = 0
    #     epe_mean = 0
    #     f1 = 0
    #     t=0

    #     self.model.eval()

    #     for i in range(200):

    #         img0, img1, flow_gt, imgAux , valid = test_dataset[i]
    #         img0, img1 = img0[None,...].cuda(), img1[None,...].cuda()
    #         imgAux = imgAux[None,...].cuda()
    #         valid = valid[None,...].cuda()
    #         flow_gt = flow_gt[None,...].cuda()
            
    #         ori_size = valid.shape[1:]
    #         inference_size = [h0,w0]

    #         image1 = F.interpolate(img0, size=inference_size, mode='bilinear',
    #                                 align_corners=True)
    #         image2 = F.interpolate(img1, size=inference_size, mode='bilinear',
    #                                 align_corners=True)
    #         scale, disp, flow, _ = self.model(image1, image2,
    #                                 attn_type=self.attn_type,
    #                                 attn_splits_list=self.attn_splits_list,
    #                                 corr_radius_list=self.corr_radius_list,
    #                                 prop_radius_list=self.prop_radius_list,
    #                                 num_reg_refine=self.num_reg_refine,
    #                                 testing=True,
    #                                 )
    #         if type(flow) == list:
    #             flow = flow[-1]
    #         if type(scale) == list:
    #             scale = scale[-1]
    #         if type(disp) == list:
    #             disp = disp[-1]

    #         if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
    #             scale = F.interpolate(scale, size=ori_size, mode='bilinear',
    #                                     align_corners=True)
    #             disp = F.interpolate(disp, size=ori_size, mode='bilinear',
    #                                     align_corners=True)
    #             flow = F.interpolate(flow, size=ori_size, mode='bilinear',
    #                                     align_corners=True)

    #             flow[:,0:1,...] = flow[:,0:1,...] * ori_size[-1] / inference_size[-1]
    #             flow[:,1:,...] = flow[:,1:,...] * ori_size[0] / inference_size[0]

    #         ls, epe, f1 = get_loss_test(scale, flow, imgAux, flow_gt, valid)

    #         total_loss += ls.item()
    #         f1_mean += f1.item()
    #         epe_mean += epe.item()
    #         t += 1

    #     loss = total_loss / float(t)
    #     f1 = f1_mean / float(t)
    #     epe = epe_mean / float(t)

    #     return loss, f1, epe, 0.

    def eva_dataloader(self, filepath):

        img_file_path = filepath
        left_fold  = 'image_2/'
        flow_noc   = 'flow_occ/'

        train_img = [img for img in os.listdir(img_file_path+left_fold) if img.find('_10') > -1]
        train_img = [i for i in train_img if int(i.split('_')[0])%5==0]
        train = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
        train = [i for i in train if int(i.split('_')[0])%5==0]

        l0_train  = [img_file_path+left_fold+img for img in train_img]
        l1_train = [img_file_path+left_fold+img.replace('_10','_11') for img in train_img]
        flow_train = [filepath+flow_noc+img for img in train]

        return sorted(l0_train), sorted(l1_train), sorted(flow_train)

    def average_gradients(self):  ##每个gpu上的梯度求平均
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data,op = dist.reduce_op.SUM)
                param.grad.data /= size

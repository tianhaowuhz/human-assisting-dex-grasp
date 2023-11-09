import os
import numpy as np
import cv2 
from tqdm import tqdm
from torch.utils.data import Dataset
from ipdb import set_trace
import torch

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def save_video(env, states, save_path, simulation=False, fps = 50, render_size = 256, suffix='avi'):
    # states: [state, ....]
    # state: (60, )
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        # set_trace()
        env_id = state[-1].long()
        env.set_states(state.unsqueeze(0))
        img = env.render(rgb=True,img_size=render_size)[env_id]
        imgs.append(img.cpu().numpy())
    if suffix == 'gif':
        from PIL import Image
        images_to_gif(save_path+f'.{suffix}', [Image.fromarray(img[:, :, ::-1], mode='RGB') for img in imgs], fps=len(imgs)//5)
    else:
        batch_imgs = np.stack(imgs, axis=0)
        images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))

def images_to_gif(path, images, fps):
    images[0].save(path, save_all=True, append_images=images[1:], fps=fps, loop=0)

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item.astype(np.uint8))
    out.release()

def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key

def ts_search():
    "ode larger t better, 1.0,    pc small t=0.05 better, 1000 step is enough for pc"
    for t in [1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]:
        for num_steps in [500,1500,2000,2500,3000]:
            for sampler in ['pc', 'ode']:
                print(t, num_steps, sampler)
                if args.relative:
                    hand_dof = dex_data[:,:18].clone().to(device).float()
                    hand_pos = dex_data[:,18:21].clone().to(device).float()
                    hand_quat = dex_data[:,21:25].clone().to(device).float()
                    obj_pcl = dex_data[:,25:3097].clone().to(device).float().reshape(-1, 1024, 3)
                    obj_pcl_2h = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, 1024)
                else:
                    hand_dof = dex_data[:,:25].clone().to(device).float()
                    obj_pcl_2h = dex_data[:,25:3097].clone().to(device).float().reshape(-1, 3, 1024)

                if sampler == 'ode' and t>0.001:
                    in_process_sample, res = cond_ode_sampler(
                        score,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        (hand_dof[:test_num], obj_pcl_2h[:test_num]),
                        t0=t,
                        device=device,
                        num_steps=num_steps,
                        batch_size=test_num,
                    )
                elif sampler == 'pc':
                    in_process_sample = pc_sampler(
                        score,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        (hand_dof[:test_num], obj_pcl_2h[:test_num]),
                        batch_size = test_num,
                        num_steps=num_steps,
                        device=device,
                        t0=t,
                    )
                    in_process_sample = in_process_sample.reshape(in_process_sample.size(0),test_num,-1)
                    res = in_process_sample[-1]

                eval_data[:,:18] = res[:,:18]
                visualize_states(eval_data, writer, 'ts_search', i, save_path=debug_path + f'{sampler}_{t}_{num_steps}')

class DexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_ot_idx = {}
        # set_trace()
        self.data_dim = self.dataset.shape[1]
        self.data_ot = {}
        obj_id = 0

        for (idx,data) in enumerate(self.dataset):
            # set_trace()
            data_id = data[3104]
            # print(data_id)
            if data_id in self.data_ot_idx:
                self.data_ot_idx[data_id].append(idx)
            else:
                self.data_ot_idx[data_id] = [idx]
                self.data_ot[obj_id] = data_id
                obj_id+=1
        
        # set_trace()
        self.data_grasp_num = np.zeros(len(self.data_ot_idx))
        for (i,data_ot_idx_each) in enumerate(self.data_ot_idx):
            # set_trace()
            self.data_grasp_num[i] = len(self.data_ot_idx[data_ot_idx_each])
        
        print('data initilized!')

    # need to overload
    def __len__(self):
        return len(self.data_ot_idx)

    # need to overload
    def __getitem__(self, idx):
        # sampled_data = np.zeros(len(idx),self.data_dim)
        # set_trace()
        sampled_idx = np.random.randint(0, self.data_grasp_num[idx])
        # print(idx,sampled_idx)
        sampled_data = self.dataset[self.data_ot_idx[self.data_ot[idx]][sampled_idx]]
        # set_trace()
        return sampled_data
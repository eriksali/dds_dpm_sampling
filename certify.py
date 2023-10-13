import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp

from runners.diffusion import Diffusion
from models.diffusion import Model
import math
import time 
import datetime 
from torchvision import transforms, datasets

from core import Smooth 
from diffmodel import DiffusionModel

import torch.nn as nn 
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torchvision.utils as tvu

from datasets import get_dataset, data_transform, inverse_data_transform



CIFAR10_DATA_DIR = "data/cifar10"

torch.set_printoptions(sci_mode=False)

import torch.nn as nn 



from transformers import AutoModelForImageClassification

 
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, default=0.5, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=10, help="the number of Monte Carlo samples for selection")
    parser.add_argument("--N", type=int, default=100, help="the number of Monte Carlo samples to use for estimation")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.0001, help="failure probability")
    parser.add_argument("--outfile", type=str, default="results/cifar10/sigma_10.5", help="output file")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--base_samples",
        type=str,
        default="fid_stats/fid_stats_cifar10_train_pytorch.npz",
        help="base samples for upsampling, *.npz",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fixed_class", type=int, default=None, help="fixed class label for conditional sampling"
    )
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed'",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpm_solver",
        help="type of dpm_solver ('dpm_solver' or 'taylor'",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--port", type=str, default="12355")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class DiffusionModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        
        args, config = parse_args_and_config()
        
        model = Model(config)
        
        diffusion = Diffusion(args, config)
        
        ##print("Loading checkpoint {}".format(ckpt))
        print("Loading checkpoint from pretrained/model-790000.ckpt")

        model.load_state_dict(
            torch.load("../_data/pretrained/cifar10/model-790000.ckpt")
            ##torch.load("pretrained/imagenet64_uncond_100M_1500K.pt")
        )
        
        model.eval()
        self.model = model 
        self.diffusion = diffusion 
        

        self.classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
        ##classifier.eval()
        '''classifier.load_state_dict(
            torch.load("pretrained/256x256_classifier.pt")
            ##torch.load("pretrained/imagenet64_uncond_100M_1500K.pt")
        )'''

        self.classifier.eval()

    def forward(self, x, t):
        x_in = x * 2 -1

        t = torch.tensor([t]*10)
        imgs = self.sampling(x_in, t[0].item())
        ##imgs = self.sampling(x, t[0].item())
        
        ##imgs = inverse_data_transform(config, x)

        imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)
        
        with torch.no_grad():
            out = self.classifier(imgs)

        imgs = inverse_data_transform(config, x)
        imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)
        
        from torchvision.utils import save_image

        output_dir = "samples/cifar10/"
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, sample in enumerate(imgs):
            save_image(sample, os.path.join(output_dir, f"sample_{i}.png"))
            
        imgs = torch.tensor(imgs).cuda()
        imgs = imgs.clone().detach()
        
        return out.logits

    
    def sampling(self, x_start, t):
        assert x_start.ndim == 4, x_start.ndim
        x0 = x_start

        ##x0 = self.scale*(x_start)
        print("x0.shape[0] in sampling is ", x0.shape[0])
        print("t in sampling is ", t)
        ##t = t[0]
    
        t_batch = torch.tensor([t] * x0.shape[0])

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        with torch.no_grad():
            out, classes = self.diffusion.sample_image(
            ##out = self.diffusion.ddim_sample(
                x_t_start,
                self.model,
                self.classifier,
                ##clip_denoised=True
            )
            
        '''with torch.no_grad():
            print("x_t_start.shape = ", x_t_start.shape)
            ##out = x_t_start,
            ##print("out.shape = ", out.shape)
            for i in range(t)[::-1]:
                ##t_batch = torch.tensor([i] * x0.shape[0]).cuda()
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    ##t,
                    clip_denoised=True
                )['pred_xstart']'''

        return out
        

    def __forward(self, x):
        ##x_in = x * 2 -1
        ##x = self.diffusion.sample(x_in, t)
        x = self.diffusion.sample(x, self.model, self.classifier)
        print("x.shape = ", x.shape)
        ##x, classes = self.diffusion.sample()
        x = torch.nn.functional.interpolate(x[0], (224, 224), mode='bicubic', antialias=True)


        img_id = 0
        for i in range(x.shape[0]):
            path = os.path.join("generated_samples/", f"{img_id}.png")
            tvu.save_image(x.cpu()[i], path)
            img_id += 1
        
        with torch.no_grad():
            self.classifier.eval()
            out = self.classifier(x * 2 -1)

        return out.logits
        
       
def certify(model, dataset, args, config):
   
    ##model = DiffusionModel()
    ##model.cuda()
    ##diffusion = Diffusion(args, config)
    
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        ##a = model.diffusion.sqrt_alphas_cumprod[t]
        ##b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        a = model.diffusion.alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod_double[t]        
        real_sigma = b / a
        ##real_sigma = model.diffusion.noise_schedule.marginal_std(t)
        ##print("a------------------------------------------------------------------")
        ##print(a)

    t = torch.tensor([t]*10)
    print(t)
    print("--------------------------------")
    
    # Define the smoothed classifier 
    smoothed_classifier = Smooth(model, 10, args.sigma, t)

    f = open(args.outfile, 'w')
    ##print("idx       \t label     \t prediction\t radius    \t correct   \t time", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    ##total_num = 0
    correct = 0
    for i in range(len(dataset)):
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()

        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        ##total_num += 1

        ##print("{}   \t\t {}\t\t\t {}\t\t\t {:.3}\t\t {}\t\t\t {}".format(
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    

def sample(args, config):

    # load model
    print('starting the model and loader...')
    model = DiffusionModel(args, config)
    model = model.eval().to(config.device)

    # load dataset
    dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    certify(model, dataset, args, config)

if __name__ == "__main__":
    args, config = parse_args_and_config()
    sample(args, config)
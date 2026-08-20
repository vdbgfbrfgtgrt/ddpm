
import os
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


IMG_EXTS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']


class ImageDataset(Dataset):
    """递归加载灰度图片；返回张量、原始尺寸 (W,H) 与相对 structure_root 的路径"""
    def __init__(self, root, structure_root=None, transform=None):
        self.root = Path(root)
        self.structure_root = Path(structure_root) if structure_root else self.root
        files = []
        for ext in IMG_EXTS:
            files.extend(self.root.rglob(ext))
        self.images = sorted(files, key=lambda p: str(p))
        assert len(self.images) > 0, f"no images found in {root}"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('L')
        orig_size = torch.tensor(img.size)  # (W, H)
        try:
            rel = path.relative_to(self.structure_root)
        except ValueError:
            rel = path.relative_to(self.root)
        if self.transform:
            img = self.transform(img)
        return img, orig_size, str(rel)


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    img_size = modelConfig["img_size"]
    dataset = ImageDataset(
        root=modelConfig["data_path"],
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                     in_channels=modelConfig["img_channels"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, _, _ in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # 无条件采样：数量/文件名/尺寸/目录结构镜像 eval_data_path 下的输入图像
    device = torch.device(modelConfig["device"])
    with torch.no_grad():
        img_size = modelConfig["img_size"]
        dataset = ImageDataset(
            root=modelConfig["eval_data_path"],
            structure_root=modelConfig["eval_structure_root"],
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["eval_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.,
                     in_channels=modelConfig["img_channels"])
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        to_pil = transforms.ToPILImage()
        out_dir = Path(modelConfig["generated_dir"])
        for images, sizes, rels in tqdm(dataloader, desc="generating"):
            noisyImage = torch.randn(
                size=[images.shape[0], modelConfig["img_channels"],
                      img_size, img_size], device=device)
            sampledImgs = sampler(noisyImage)
            sampledImgs = torch.clamp(sampledImgs * 0.5 + 0.5, 0, 1).cpu()
            for img, size, rel in zip(sampledImgs, sizes, rels):
                save_path = out_dir / rel
                save_path.parent.mkdir(parents=True, exist_ok=True)
                to_pil(img).resize((int(size[0]), int(size[1])), Image.BILINEAR).save(save_path)
        print(f"generated images saved to {out_dir}")

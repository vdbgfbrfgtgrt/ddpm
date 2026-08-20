
import os
import random
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from PIL import Image, ImageOps
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler


IMG_EXTS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']


def _list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: p.name)


class PairedImageDataset(Dataset):
    """从 root/HQ 与 root/LQ 按同名文件加载配对图像，随机翻转对 HQ/LQ 同步生效"""
    def __init__(self, root, transform=None):
        self.hq_dir = Path(root) / 'HQ'
        self.lq_dir = Path(root) / 'LQ'
        self.hq_images = _list_images(self.hq_dir)
        self.lq_images = _list_images(self.lq_dir)
        assert len(self.hq_images) > 0, f"no images found in {self.hq_dir}"
        assert len(self.hq_images) == len(self.lq_images), \
            f"HQ/LQ 数量不一致: HQ={len(self.hq_images)}, LQ={len(self.lq_images)}"
        for hq, lq in zip(self.hq_images, self.lq_images):
            assert hq.name == lq.name, f"HQ/LQ 文件名不配对: {hq.name} vs {lq.name}"
        self.transform = transform

    def __len__(self):
        return len(self.hq_images)

    def __getitem__(self, idx):
        hq = Image.open(self.hq_images[idx]).convert('L')
        lq = Image.open(self.lq_images[idx]).convert('L')
        if random.random() < 0.5:
            hq = ImageOps.mirror(hq)
            lq = ImageOps.mirror(lq)
        if self.transform:
            hq = self.transform(hq)
            lq = self.transform(lq)
        return hq, lq


class LQImageDataset(Dataset):
    """推理用：递归加载 LQ 目录下的图像，返回张量、原始尺寸与相对路径（用于镜像目录结构）"""
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
    # dataset: root 下需包含 HQ/ 与 LQ/
    img_size = modelConfig["img_size"]
    dataset = PairedImageDataset(
        root=modelConfig["data_path"],
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup: 输入 = x_t(1ch) + 条件LQ(1ch)，输出 = 噪声(1ch)
    img_channels = modelConfig["img_channels"]
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                     in_channels=img_channels * 2, out_channels=img_channels).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for hq, lq in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = hq.to(device)
                cond = lq.to(device)
                loss = trainer(x_0, cond).sum() / 1000.
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
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate: 读取 LQ 目录逐批复原，按原文件名/原始尺寸保存，目录结构镜像输入
    device = torch.device(modelConfig["device"])
    with torch.no_grad():
        img_size = modelConfig["img_size"]
        dataset = LQImageDataset(
            root=modelConfig["eval_lq_path"],
            structure_root=modelConfig["eval_structure_root"],
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["eval_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

        img_channels = modelConfig["img_channels"]
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.,
                     in_channels=img_channels * 2, out_channels=img_channels).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        to_pil = transforms.ToPILImage()
        out_dir = Path(modelConfig["restored_dir"])
        for lq, sizes, rels in tqdm(dataloader, desc="restoring"):
            cond = lq.to(device)
            noisyImage = torch.randn(
                size=[cond.shape[0], img_channels, img_size, img_size], device=device)
            restored = sampler(noisyImage, cond)
            restored = torch.clamp(restored * 0.5 + 0.5, 0, 1).cpu()
            for img, size, rel in zip(restored, sizes, rels):
                save_path = out_dir / rel
                save_path.parent.mkdir(parents=True, exist_ok=True)
                to_pil(img).resize((int(size[0]), int(size[1])), Image.BILINEAR).save(save_path)
        print(f"restored images saved to {out_dir}")

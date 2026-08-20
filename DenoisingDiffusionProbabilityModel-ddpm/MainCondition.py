from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 10,
        "eval_batch_size": 8,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 256,
        "img_channels": 1,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        # 训练配对数据：目录下需包含 HQ/ 与 LQ/（同名 png 配对）
        "data_path": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/train",
        # 复原推理输入目录（换 val/LQ 即可评估验证集）
        "eval_lq_path": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/test/LQ",
        # 输出目录结构镜像该根目录（输出为 restored_dir/test/LQ/同名.png）
        "eval_structure_root": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron",
        "training_load_weight": None,
        "save_dir": "./CheckpointsCondition/",
        "test_load_weight": "ckpt_199_.pt",
        "restored_dir": "./SampledImgs/Restored/",
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()

from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "eval_batch_size": 8,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
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
        "data_path": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/train/LQ",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        # 采样时镜像该目录：生成的数量、文件名、原始尺寸、目录结构与其一致
        "eval_data_path": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/train/LQ",
        "eval_structure_root": "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron",
        "generated_dir": "./SampledImgs/Generated/",
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()

from argparse import Namespace
from typing import Type

import torch

from nerf.gui import NeRFGUI
from nerf.network_base import BaseNeRFNetwork
from nerf.provider import NeRFDataset
from nerf.utils import *


def perform_disentanglement(network_class: Type[BaseNeRFNetwork], opt: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = NeRFDataset(opt, device=device, type="test").dataloader()

    print("LOADING MODELS...")
    get_latest_checkpoint = lambda dir: sorted(
        glob.glob(os.path.join(opt.full, "checkpoints/ngp.pth.tar"))
    )[-1]

    model_full = network_class(
        encoding="hashgrid", bound=opt.bound, cuda_ray=opt.cuda_ray, density_scale=1
    )
    model_full.load_state_dict(
        torch.load(get_latest_checkpoint(opt.full), map_location=device)["model"]
    )
    model_full.eval()

    model_bg = network_class(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=10 if opt.mode == "blender" else 1,
    )
    model_bg.load_state_dict(
        torch.load(get_latest_checkpoint(opt.bg), map_location=device)["model"]
    )
    model_bg.eval()
    print("LOADED MODELS!")

    trainer = Trainer(
        "ngp",
        opt,
        model=model_full,
        background_model=model_bg,
        device=device,
        criterion=torch.nn.MSELoss(reduction="none"),
        fp16=opt.fp16,
        metrics=[PSNRMeter()],
    )
    trainer.test(test_loader)

    return

    save_path = os.path.join(opt.full, "results")

    with torch.no_grad():
        # TODO: maybe enable the following at some point?
        # with torch.cuda.amp.autocast(enabled=opt.fp16):
        # model_full.update_extra_state()
        # model_bg.update_extra_state()

        for i, data in enumerate(test_loader):
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                rays_o = data["rays_o"]  # [B, N, 3]
                rays_d = data["rays_d"]  # [B, N, 3]
                H, W = data["H"], data["W"]

                print("RENDER", rays_o.device, rays_d.device)
                outputs = model_full.render(
                    rays_o, rays_d, staged=True, bg_color=None, perturb=False, **vars(opt)
                )

                preds = outputs["image"].reshape(-1, H, W, 3)
                # preds_depth = outputs["depth"].reshape(-1, H, W)

                path = os.path.join(save_path, f"{i:04d}.png")
                cv2.imwrite(
                    path,
                    cv2.cvtColor(
                        (preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                    ),
                )

            break

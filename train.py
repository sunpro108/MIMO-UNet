import os
import torch
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange

from data import train_dataloader, valid_dataloader
from utils import Adder, Timer, check_lr


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(
        subset=args.subset,
        archive=args.archive,
        use_subarch=args.use_subarch,
        batch_size=args.batch_size,
        num_workers=args.num_worker
    )
    model.eval()
    psnr_adder = Adder()
    dwt = DWTForward(J=3, wave='haar', mode='zero').to(device)
    iwt = DWTInverse(wave='haar', mode='zero').to(device)
    psnr1 = PSNR(data_range  = (0.0, 255.0), reduction = "elementwise_mean", dim = (1, 2, 3))
    psnr2 = PSNR(data_range  = (0.0, 255.0), reduction = "elementwise_mean", dim = (1, 2, 3))
    psnr3 = PSNR(data_range  = (0.0, 255.0), reduction = "elementwise_mean", dim = (1, 2, 3))

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img, mask = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            lc, x = dwt(input_img)
            lr, h = dwt(label_img)
            # if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
            #     os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(x)
            # ic = iwt((lc, x))
            ip = iwt((lr, pred))
            # ip = iwt((lc, pred))

            label_img = (255 * torch.clip(0.5 * label_img + 0.5, 0, 1)).to(torch.uint8)

            ip = 0.5 * ip + 0.5
            ip = torch.clamp(ip, 0, 1)
            ip = (255*ip).to(torch.uint8)
            p1 = psnr1.forward(ip, label_img)

            # ir = 0.5 * ir + 0.5
            # ir = torch.clamp(ir, 0, 1)
            # ir = (255*ir).to(torch.uint8)
            # p2 = psnr2.forward(ir, label_img)
            

            psnr_adder(p1.cpu().numpy())
            # psnr_adder(p2.cpu().numpy())
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(
        subset=args.subset,
        archive=args.archive,
        use_subarch=args.use_subarch,
        batch_size=args.batch_size,
        num_workers=args.num_worker
    )
    max_iter = len(dataloader)
    print(max_iter)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    dwt = DWTForward(J=3, wave='haar', mode='zero').to(device)
    iwt = DWTInverse(wave='haar', mode='zero').to(device)
    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img, mask = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            _, x = dwt(input_img)
            label_ll, label = dwt(label_img)

            optimizer.zero_grad()
            pred = model(x)
            l1 = criterion(pred[0], label[0])
            l2 = criterion(pred[1], label[1])
            l3 = criterion(pred[2], label[2])
            loss_content = l1+l2+l3

            img_pred = iwt((label_ll, pred))
            loss_fft = criterion(img_pred, label_img)

            if iter_idx % 100 == 0:
                img_pred_show = (255 * torch.clamp(0.5 * img_pred.detach().clone() + 0.5, 0, 1)).to(torch.uint8)
                img_pred_show = rearrange(img_pred_show, 'b c h w -> (b h) w c')
                Image.fromarray(
                    img_pred_show.cpu().numpy()
                ).save(f'results/{args.model_path}/result_image/img_{iter_idx}.png')

            # img_label_show = (255 * torch.clamp(0.5 * label_img + 0.5, 0, 1)).to(torch.uint8)
            # label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft1 = torch.rfft(pred_img[0], signal_ndim=2, normalized=False, onesided=False)
            # label_fft2 = torch.rfft(label_img2, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft2 = torch.rfft(pred_img[1], signal_ndim=2, normalized=False, onesided=False)
            # label_fft3 = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft3 = torch.rfft(pred_img[2], signal_ndim=2, normalized=False, onesided=False)

            # f1 = criterion(pred_fft1, label_fft1)
            # f2 = criterion(pred_fft2, label_fft2)
            # f3 = criterion(pred_fft3, label_fft3)
            # loss_fft = f1+f2+f3

            loss = loss_content + 1.0 * loss_fft
            loss = loss_content 
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR_GOPRO', val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)

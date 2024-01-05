from torch.utils.data import DataLoader 
import torch 
import os 
import os.path as osp
import scipy.io as scio
import numpy as np 
import einops
from opts import parse_args
from model.saunet import DUNet_plus
from utils import Logger, load_checkpoint, TestData, compare_ssim, compare_psnr
import time
import cv2
import math
from skimage.metrics import structural_similarity as ski_ssim



def test(args, network, logger, test_dir, writer=None, epoch=1):
    network = network.eval()
    test_data = TestData(args) 
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)    

    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    rec_list,gt_list = [],[]
    for iter, data in enumerate(test_data_loader):
        gt = data 
        gt = gt[0].float().numpy()
        pic = gt.copy()

        if gt.shape[0] > gt.shape[1]:
            rotated = True
            inp = cv2.rotate(gt[:,:,0], cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = False
            inp = gt[:,:,0]
        if inp.shape[0] != args.size[0] or inp.shape[1] != args.size[1]:
            raise ValueError('The size of test image is not same with the trained size.')

        with torch.no_grad():
            out, _, _, _, _ = network(torch.from_numpy(inp).unsqueeze(0).to(args.device))

        out = out.squeeze().cpu().numpy()
        out = np.clip(out,0,1)
        psnr = compare_psnr(inp*255,out*255)
        # ssim = compare_ssim(inp*255,out*255)
        ssim = ski_ssim(inp*255,out*255,data_range=255)
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))
        if rotated:
            pic[:,:,0] = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            pic[:,:,0] = out
        
        rec_list.append(pic)
        gt_list.append(gt)


    for i,name in enumerate(test_data.data_list):
        _name,_ = name.split(".")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        image_name = os.path.join(test_dir, _name+"_"+"epoch_"+str(epoch)+".png")
        gt_rgb = cv2.cvtColor(gt_list[i], cv2.COLOR_YCrCb2BGR)
        rec_rgb = cv2.cvtColor(rec_list[i], cv2.COLOR_YCrCb2BGR)
        result_img = np.concatenate([gt_rgb,rec_rgb],axis=1)*255
        result_img = result_img.astype(np.float32)  
        cv2.imwrite(image_name,result_img)

    if writer is not None:
        writer.add_scalar("psnr_mean",np.mean(psnr_list),epoch)
        writer.add_scalar("ssim_mean",np.mean(ssim_list),epoch)
    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    torch.set_float32_matmul_precision('highest')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    args = parse_args()
    # args.test_weight_path = '/home/wangping/codes/SAUNet/checkpoint/cbsd68/cr_50_epoch_78.pth'
    # args.cr = 0.50
    # args.size = [321,481]
    # args.meas_size = [227,340]     #[32,48] [64,96] [102,152] [161,240] [227,340]
    test_path = "test_results" + "/" + "cr_" + str(args.cr)
    if not os.path.exists(test_path):
        os.makedirs(test_path,exist_ok=True)
    network = DUNet_plus(imag_size=args.size, 
                    meas_size= args.meas_size,
                    img_channels=args.color_channels,
                    channels=args.channels,
                    mid_blocks=args.mid_blocks,
                    enc_blocks=args.enc_blocks,
                    dec_blocks=args.dec_blocks,
                    stages=args.stages,
                    matrix_train = args.matrix_train).to(args.device)

    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        logger.info('Loading pretrained model...')
        pretrained_dict = torch.load(args.test_weight_path) 
        load_checkpoint(network, pretrained_dict)
    else:
        raise ValueError('Please input a weight path for testing.')
    psnr_dict, ssim_dict = test(args, network, logger, test_path)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))


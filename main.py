import argparse, torch, random, os
import numpy as np
import torch.nn.functional as F
from utils import dataset_utils, model_utils, plot_utils
from approx_hessian import approx_surface
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main(args):
    save_path = f"{args.save_path}{args.model_path.replace('/', '_')}"
    save_settings = f'dg{args.dg}_dr{args.dr}_n{args.save_nimg}'
    flat_settings = f'n{args.flat_nimg}'

    # fixing seed
    seed = 706
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU setting
    device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    device_ids.sort()
    
    if torch.cuda.is_available() and len(device_ids)>0:
        device = torch.device(f'cuda:{device_ids[0]}')
        if torch.cuda.device_count() <= device_ids[-1]:
            raise ValueError(f"The maximum possible gpu id is {torch.cuda.device_count()-1}, but now entered {device_ids[-1]}.")
    else:
        device = torch.device('cpu')
    
    # Model setting
    model = model_utils.set_model(args.model, args.model_path)
    model = model.to(device) if len(device_ids)<2 else torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    model.eval()

    # DataLoader setting
    save_loader, flatness_loader, save_indices, flatness_indices = dataset_utils.set_dataloader(args.dataset, args.data_path, args.save_nimg, args.flat_nimg, args.batch_size)
    map_dataloader, glist, rlist = dataset_utils.map_dataloader(model, save_loader, args.batch_size, device, args.dg, args.dr)

    if not os.path.exists(f'{save_path}/losses_{save_settings}.pt'):
        # Calculate Loss landscape values
        losses, correct = None, None
        with torch.no_grad():
            for x, y in tqdm(map_dataloader, desc='loss landscape'):
                x, y = x.to(device), y.to(device).view(-1)
                logit = model(x)
                pred = F.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                loss = F.cross_entropy(logit, y, reduction='none')
                losses = torch.cat((losses, loss)) if losses is not None else loss
                correct = torch.cat((correct, outputs==y)) if correct is not None else (outputs==y)
        losses, correct = losses.reshape(args.nimg, glist[2], rlist[2]).cpu(), correct.reshape(args.nimg, glist[2], rlist[2]).cpu()

        # save file and plots
        os.makedirs(f'{save_path}/', exist_ok=True)
        torch.save(save_indices, f'{save_path}/save_indices_{save_settings}.pt')
        torch.save(losses, f'{save_path}/losses_{save_settings}.pt')
        torch.save(correct, f'{save_path}/correct_{save_settings}.pt')
        
    if args.hessian and not os.path.exists(f'{save_path}/eigenvalue_{flat_settings}.pt'):
        # calculate hessian
        H = plot_utils.Hessian(model, device)
        eigenvalues = {}
        for idx, (x, y) in zip(tqdm(flatness_indices), flatness_loader):
            x, y = x.to(device), y.to(device)
            x.requires_grad = True
            H.set_y(y)
            hessian = torch.autograd.functional.hessian(H.get_hessian, x.reshape(-1), create_graph=True)
            eigenvalue = H.get_eigenvalue(hessian)
            eigenvalues[idx.item()] = eigenvalue.item()
        
        os.makedirs(f'{save_path}/', exist_ok=True)
        torch.save(eigenvalues, f'{save_path}/eigenvalue_{flat_settings}.pt')
    
    if args.plot:
        indices = torch.load(f'{save_path}/indices_{save_settings}.pt').numpy()
        losses = torch.load(f'{save_path}/losses_{save_settings}.pt').numpy()
        correct = torch.load(f'{save_path}/correct_{save_settings}.pt').numpy()

        gmin, gmax, gnum = [eval(x) for x in args.dg.split(":")]
        rmin, rmax, rnum = [eval(y) for y in args.dr.split(":")]
        dg = np.linspace(gmin, gmax, gnum)
        dr = np.linspace(rmin, rmax, rnum)

        for idx, loss, cor in zip(indices, losses, correct):
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            X, Y = np.meshgrid(dg, dr)
            # surf = ax.plot_surface(X, Y, loss, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
            #                     linewidth=0, antialiased=False)
            surf = ax.plot_surface(X, Y, loss, cmap=args.cmap, linewidth=0, antialiased=False)

            ax.set_xlim(ax.get_xlim()[::-1])
            # ax.set_zlim(args.vmin, args.vmax)
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.plot_wireframe(X, Y, loss, color='black', linewidth=0.2)

            plt.title(f'{args.model_path.replace("/", "_")}\nloss #{idx}')
            fig.colorbar(surf, shrink=0.5, aspect=5, cmap=args.cmap)

            plt.savefig(f'{save_path}/losses_{idx}.png')

            if args.classification_plot:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

                X, Y = np.meshgrid(dg, dr)
                cmap = ListedColormap([plt.get_cmap(args.cmap)(170), plt.get_cmap(args.cmap)(85)])

                colors = cmap(cor)
                # surf = ax.plot_surface(X, Y, loss, facecolors=colors, vmin=args.vmin, vmax=args.vmax, linewidth=0, antialiased=False)
                surf = ax.plot_surface(X, Y, loss, facecolors=colors, linewidth=0, antialiased=False)

                ax.set_xlim(ax.get_xlim()[::-1])
                # ax.set_zlim(args.vmin, args.vmax)
                ax.plot_wireframe(X, Y, loss, color='black', linewidth=0.2)

                ax.zaxis.set_major_formatter('{x:.02f}')

                plt.title(f'{args.model_path.replace("/", "_")}\ncorrect #{idx}')

                blue_patch = plt.Line2D([0], [0], color=plt.get_cmap(args.cmap)(170), lw=4, label='Wrong')
                white_patch = plt.Line2D([0], [0], color=plt.get_cmap(args.cmap)(85), lw=4, label='Correct')
                ax.legend(handles=[blue_patch, white_patch])

                plt.savefig(f'{save_path}/corrects_{idx}.png')

            plt.close()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', default='0', type=str, help='list of gpu ids to use, split into ","')
    
    # model settings
    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--model_path', default='models/no_attack.pt', help='relative path of model from main.py')
    parser.add_argument('--save_path', default='results/', help='relative path of directory from main.py')
    
    # data settings
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--data_path', default='data/', help='relative path of dataset')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--dg', default='0:0.0314:51', help='a string with format xmin:xmax:xnum, gradient direction')
    parser.add_argument('--dr', default='0:0.0314:51', help='a string with format ymin:ymax:ynum, random direction')

    # plot settings
    parser.add_argument('--plot', action='store_true', default=False, help='save matplotlib 3D plot')
    parser.add_argument('--classification_plot', action='store_true', default=False, help='coloring according to the correct answer')
    parser.add_argument('--save_nimg', default=10, type=int, help='number of images to save plots')
    parser.add_argument('--vmax', default=3.5, type=float, help='maximum value to map')
    parser.add_argument('--vmin', default=2.0, type=float, help='minimum value to map')
    parser.add_argument('--cmap', default='coolwarm', help='color maps in matplotlib')

    # flatness measurement settings
    parser.add_argument('--hessian', action='store_true', default=False, help='measure ')
    parser.add_argument('--flat_nimg', default=100, type=int, help='number of images to calculate mean and std of hessian matrix dominant eigenvalue')
    
    # parser.add_argument('--vtp') #TODO: use ParaView

    args = parser.parse_args()
    main(args)

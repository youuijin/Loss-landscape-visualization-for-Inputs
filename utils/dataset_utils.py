import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def set_dataloader(dataset, path, nimg, batch_size):
    dataset = dataset.lower()

    # Image transform setting
    # ** Set the same as your learning environment **
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    if dataset == 'cifar10':
        data = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        data = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)
    
    indices = torch.randperm(len(data))[:nimg]
    sampler = SubsetRandomSampler(indices)

    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)

    return dataloader, indices

def map_dataloader(model, dataloader, batch_size, device, dg, dr):
    gmin, gmax, gnum = [eval(x) for x in dg.split(":")]
    rmin, rmax, rnum = [eval(y) for y in dr.split(":")]
    gvalues = torch.linspace(gmin, gmax, gnum).view(gnum, 1, 1, 1, 1).to(device)
    rvalues = torch.linspace(rmin, rmax, rnum).view(1, rnum, 1, 1, 1).to(device)
    
    new_data, new_y = None, None

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()

        x.grad.sign_()
        grad_direction = x.grad
        rand_direction = (torch.randint(0, 2, size=x.shape)*2-1).to(device)

        result_tensor = x.unsqueeze(1).unsqueeze(2) + gvalues * grad_direction.unsqueeze(1).unsqueeze(2) \
            + rvalues * rand_direction.unsqueeze(1).unsqueeze(2)
        result_tensor = result_tensor.view(-1, x.shape[1], x.shape[2], x.shape[3])

        if new_data == None:
            new_data = result_tensor
            new_y = torch.repeat_interleave(y, gnum*rnum)
        else:
            new_data = torch.cat((new_data, result_tensor), dim=0)
            new_y = torch.cat((new_y, torch.repeat_interleave(y, gnum*rnum)), dim=0)

    new_data = TensorDataset(new_data, new_y.unsqueeze(1))
    map_dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=False)

    return map_dataloader, (gmin, gmax, gnum), (rmin, rmax, rnum)
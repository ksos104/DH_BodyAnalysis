import torch
import time
from tqdm import tqdm
import networks

def main():
    model = networks.init_model('resnet18', 7, pretrained=None)

    state_dict = torch.load('./log/checkpoint_100.pth.tar')['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    avg_time = 0.0
    for _ in tqdm(range(1000)):
        sample = torch.rand([1, 3, 2048, 1024]).cuda()
        start = time.time()
        model(sample)
        end = time.time()
        avg_time = avg_time + (end - start)
    
    avg_time = avg_time / 1000
    print("avg_time: %.4f sec" % avg_time)

if __name__ == '__main__':
    main()
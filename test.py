from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from regretnet.datasets import generate_dataset_1x2, generate_dataset_nxk
from regretnet.regretnet import RegretNet, train_loop, test_loop
from torch.utils.tensorboard import SummaryWriter
from regretnet.datasets import Dataloader
from util import plot_12_model, plot_payment, plot_loss, plot_regret
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=3000)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('--misreport-lr', type=float, default=2e-2)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--p_activation', default="")
parser.add_argument('--model', default="")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    checkpoint = torch.load(args.model)
    print("Architecture:")
    print(json.dumps(checkpoint['arch'], indent=4, sort_keys=True))
    print("Training Args:")
    print(json.dumps(vars(checkpoint['args']), indent=4, sort_keys=True))

    #override p_activation
    if args.p_activation != "":
        checkpoint['arch']['p_activation'] = args.p_activation


    model = RegretNet(**checkpoint['arch']).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    test_data = generate_dataset_nxk(checkpoint['arch']['n_agents'],
                                     checkpoint['arch']['n_items'], args.test_num_examples).to(DEVICE)
    args.n_agents = checkpoint['arch']['n_agents']
    args.n_items = checkpoint['arch']['n_items']
    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    result = test_loop(model, test_loader, args, device=DEVICE)
    print(f"Experiment:{checkpoint['name']}")
    print(json.dumps(result, indent=4, sort_keys=True))

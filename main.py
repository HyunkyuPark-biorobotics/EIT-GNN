from argparse import ArgumentParser
import os
import time
import torch
import torch.optim as optim
import numpy as np
from network import ModelGen
from utils import get_dataset, EarlyStopping, WeightedMSELoss
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import thop

scaler = torch.cuda.amp.GradScaler()


def cli_main(args):
    device = 'cuda' if args.cuda else 'cpu'

    # initialize logger
    # exp_path = os.path.join('pytorch_results', args.model_type)
    exp_path = os.path.join('pytorch_results', args.name)
    os.makedirs(exp_path, exist_ok=True)
    with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
        f.write(str(vars(args)))

    # initialize dataset and model
    trn, val, tst, tst_raw, cfg = get_dataset(args, 'None')


    batch_trn = DataLoader(trn, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batch_val = DataLoader(val, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = ModelGen(cfg, args)

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=20)

    criterion = WeightedMSELoss(weights=cfg['volume_weight'].cuda(), adj_mat=cfg['adj_mat_out'].cuda())

    val_losses = []

    scalar_op = False
    grad_accum = 1  # Due to the memory limit, 128 * 2 batch size is emulated by grad_accum.
    time_start = time.time()
    for step in range(1, args.n_steps + 1):
        trn_loss_val = 0.0
        optimizer.zero_grad()
        for iter, (b_trn_in, b_trn_out) in enumerate(batch_trn):
            if scalar_op:
                # with torch.cuda.amp.autocast():  # cast mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    trn_preds = model(b_trn_in.cuda())
                    trn_loss_recon, trn_loss_lap = criterion(trn_preds, b_trn_out.cuda()) / grad_accum
                    trn_loss = trn_loss_recon + args.hp * trn_loss_lap
                  
                scaler.scale(trn_loss).backward()  # scaled gradients
                if (iter + 1) % grad_accum == 0:  # Wait for several backward steps
                    scaler.step(optimizer)  # unscaled gradients
                    scaler.update()  # scaled update
                    optimizer.zero_grad()
                    trn_loss_val += trn_loss.detach().item()

            else:
                trn_preds = model(b_trn_in.cuda())
                trn_loss_recon, trn_loss_lap = criterion(trn_preds, b_trn_out.cuda()) #/ grad_accum
                trn_loss = trn_loss_recon + args.hp * trn_loss_lap

                trn_loss.backward()
                if (iter + 1) % grad_accum == 0:  # Wait for several backward steps
                    optimizer.step()  # Now we can do an optimizer step
                    optimizer.zero_grad()
                    trn_loss_val += trn_loss.detach().item()
        # print(iter)

        trn_loss_val /= (iter + 1)

        model.eval()

        print('Iter {:04d} | Train Loss recon {:.6f} | {:d}s elapsed'.format(
            step, trn_loss_val, int(time.time() - time_start)))

        # evaluate validation set
        with torch.no_grad():
            val_loss_val = 0.0
            for iter, (b_val_in, b_val_out) in enumerate(batch_val):
                # print(bg)

                val_preds = model(b_val_in.cuda())
                val_loss_recon, val_loss_lap = criterion(val_preds, b_val_out.cuda())
                val_loss = val_loss_recon + args.hp * val_loss_lap

                val_loss_val += val_loss.detach().item()
            # print(iter)
            val_loss_val /= (iter + 1)

        val_losses.append(val_loss_val)
        print('Iter {:04d} | Test Loss {:.6f}'.format(
            step, val_loss_val))

        early_stopping(val_loss_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        # save models and plots
        if step % args.save_step == 0:
            print('Saving models...', end='', flush=True)

            torch.save(model.state_dict(), os.path.join(exp_path, f"model_{step}.pth"))
            torch.save(early_stopping.best_model.state_dict(), os.path.join(exp_path, f"model_best.pth"))

            print(' {}s elapsed'.format(int(time.time() - time_start)))

        time_start = time.time()
        model.train()

    # After training done
    np.savetxt(os.path.join(exp_path, "val_loss.txt"), early_stopping.loss_history, delimiter=",")
    torch.save(early_stopping.best_model.state_dict(), os.path.join(exp_path, f"model_best.pth"))
    print("The lowest validation loss is : " + str(early_stopping.best_loss))
    np.savetxt(os.path.join(exp_path, f"minimum_val_loss.txt"),
               np.vstack((early_stopping.best_loss, early_stopping.best_loss)), delimiter=",")

    # test
    model = early_stopping.best_model
    batch_tst = DataLoader(tst, batch_size=args.batch_size, shuffle=False, drop_last=True)

    with torch.no_grad():
        tst_loss_val_recon = 0.0
        tst_loss_val_lap = 0.0
        for iter, (b_tst_in, b_tst_out) in enumerate(batch_tst):
            tst_preds = model(b_tst_in.cuda())
            tst_loss_recon, tst_loss_lap = criterion(tst_preds, b_tst_out.cuda())
            tst_loss_val_recon += tst_loss_recon.detach().item()
            tst_loss_val_lap += tst_loss_lap.detach().item()

        tst_loss_val_recon /= (iter + 1)
        tst_loss_val_lap /= (iter + 1)
    print('Test Loss {:.6f}'.format(
        tst_loss_val_recon))
    tst_loss_val = tst_loss_val_recon + args.hp * tst_loss_val_lap

    np.savetxt(os.path.join(exp_path, f"test_loss_SSAMSE.txt"), np.vstack((tst_loss_val, tst_loss_val)), delimiter=",")
    np.savetxt(os.path.join(exp_path, f"test_loss_MSE.txt"), np.vstack((tst_loss_val_recon, tst_loss_val_recon)), delimiter=",")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='sample_trial', help='experiment name')
    parser.add_argument('--model_type',
                        choices=['MLP_MLP', 'TF_MLP', 'MLP_CNN', 'TF_CNN', 'MLP_GCN', 'OGN_GCN', 'TF_GCN'],
                        type=str, default='TF_GCN', help='model type')
    parser.add_argument('--mesh_type', choices=['face'], type=str, default='face',
                        help='model type')
    parser.add_argument('--hp', type=float, default=1e0, help='Loss hyperparameter for laplacian')

    # TF-GCN model parameters
    parser.add_argument('--TF_layers', type=int, default=3, help='number of TF layers')
    parser.add_argument('--GCN_layers', type=int, default=3, help='number of GCN layers')
    parser.add_argument('--feature_num', type=int, default=32, help='number of features')
    parser.add_argument('--readout', choices=['MLP'], type=str, default='MLP', help='readout type')

    # training hyperparams
    parser.add_argument('--noise', type=float, default=0.9e-3, help='Gaussian noise std.')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--n_steps', type=int, default=20000, help='number of steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dataset_ratio', type=float, default=1.0, help='ratio of dataset for learning')
    parser.add_argument('--trn_ratio', type=float, default=0.8, help='ratio of dataset training for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of dataset validation for training')

    # misc hyperparams
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--print_step', type=int, default=1, help='print step')
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--save_step', type=int, default=10, help='save step')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args.name = args.name + "_" + args.model_type + "_" + args.mesh_type

    print(args.name)
    cli_main(args)

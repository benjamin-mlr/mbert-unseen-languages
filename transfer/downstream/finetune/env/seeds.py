from transfer.downstream.finetune.env.imports import sys, np, torch

def init_seed(args, verbose=0):
    if verbose:
        print("IMPORTS : initializing seeds...")
    sys.path.insert(0, ".")
    # SEED_TORCH used for any model related randomness + batch picking, dropouts, ..
    # SEED_NP used for picking the bucket, for generating word embedding when loading embedding matrix and maybe other stuff
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

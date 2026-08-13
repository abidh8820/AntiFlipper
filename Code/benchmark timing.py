"""
benchmark_timing.py

Quick real-hardware timing benchmark for AntiFLipper experiments.

WHAT THIS DOES
---------------
Runs two short training sessions per dataset (a "short" one and a "long" one,
differing only in global_rounds) and uses the time DIFFERENCE between them to
back out a clean per-round cost, separate from one-time setup cost (data
loading, model construction, etc). This is more accurate than just timing one
run and dividing by its round count, because setup cost would otherwise get
smeared across the per-round estimate.

HOW TO RUN
----------
1. Copy this file into your Code/ directory (next to environment_federated.py).
2. Make sure you can already run your existing notebooks successfully (same
   environment/GPU).
3. From a terminal in that directory: `python benchmark_timing.py`
4. It will print an estimated setup cost and per-round cost for MNIST and
   CIFAR-10, plus a rough total-time projection for your real round counts.

WHAT TO DO WITH THE OUTPUT
---------------------------
Send me the printed numbers (or just the final "Summary" block) and I'll
recompute the real experiment schedule instead of the rough estimates I gave
you earlier.

NOTE: Edit the config blocks below (NUM_PEERS, LOCAL_LR, etc.) if your actual
notebook configs differ from what's currently in Experiments_MNIST.ipynb /
Experiments_CIFAR10.ipynb -- match them exactly for an accurate estimate.
"""
import time
import torch
from torch import nn
from experiment_federated import run_exp


def time_run(dataset_name, model_name, dd_type, num_peers, frac_peers,
             global_rounds, local_epochs, local_bs, local_lr, local_momentum,
             labels_dict, device, attackers_ratio, alpha, source_class, target_class):
    start = time.time()
    run_exp(
        dataset_name=dataset_name, model_name=model_name, dd_type=dd_type,
        num_peers=num_peers, frac_peers=frac_peers, seed=0,
        test_batch_size=1000, criterion=nn.CrossEntropyLoss(),
        global_rounds=global_rounds, local_epochs=local_epochs,
        local_bs=local_bs, local_lr=local_lr, local_momentum=local_momentum,
        labels_dict=labels_dict, device=device, attackers_ratio=attackers_ratio,
        attack_type='label_flipping', malicious_behavior_rate=1,
        rule='localEval', class_per_peer=2, samples_per_class=250,
        rate_unbalance=1, alpha=alpha, source_class=source_class,
        target_class=target_class, resume=False,
    )
    return time.time() - start


def estimate_per_round(dataset_name, model_name, num_peers, frac_peers,
                        local_epochs, local_bs, local_lr, local_momentum,
                        labels_dict, device, dd_type,
                        attackers_ratio=0.4, alpha=1,
                        source_class=0, target_class=1,
                        short_rounds=2, long_rounds=6):
    print(f"\n=== Benchmarking {dataset_name} ({dd_type}, {num_peers} peers, "
          f"frac={frac_peers}) ===")
    print(f"Running a {short_rounds}-round warm-up session...")
    t_short = time_run(dataset_name, model_name, dd_type, num_peers, frac_peers,
                        short_rounds, local_epochs, local_bs, local_lr,
                        local_momentum, labels_dict, device, attackers_ratio,
                        alpha, source_class, target_class)

    print(f"Running a {long_rounds}-round session...")
    t_long = time_run(dataset_name, model_name, dd_type, num_peers, frac_peers,
                       long_rounds, local_epochs, local_bs, local_lr,
                       local_momentum, labels_dict, device, attackers_ratio,
                       alpha, source_class, target_class)

    per_round = (t_long - t_short) / (long_rounds - short_rounds)
    setup_cost = t_short - short_rounds * per_round
    print(f"--> Estimated one-time setup cost: {setup_cost:.1f}s")
    print(f"--> Estimated per-round cost:      {per_round:.1f}s")
    return per_round, setup_cost


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- MNIST config: match Experiments_MNIST.ipynb exactly ----
    mnist_labels = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4,
                     'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
    per_round_mnist, setup_mnist = estimate_per_round(
        dataset_name='MNIST', model_name='CNNMNIST',
        num_peers=100, frac_peers=1,
        local_epochs=3, local_bs=64, local_lr=0.001, local_momentum=0.9,
        labels_dict=mnist_labels, device=device, dd_type='NON_IID',
    )

    # ---- CIFAR-10 config: EDIT this once you've resolved the NUM_PEERS
    # discrepancy from the methodology audit (paper says 100/20 active;
    # notebook currently has NUM_PEERS=20, frac_peers=1) ----
    cifar_labels = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4,
                     'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
    per_round_cifar, setup_cifar = estimate_per_round(
        dataset_name='CIFAR10', model_name='ResNet18',
        num_peers=20, frac_peers=1,
        local_epochs=3, local_bs=32, local_lr=0.01, local_momentum=0.9,
        labels_dict=cifar_labels, device=device, dd_type='IID',
    )

    print("\n" + "=" * 50)
    print("SUMMARY -- send me this block")
    print("=" * 50)
    print(f"MNIST:    ~{per_round_mnist:.1f}s/round "
          f"-> 200 rounds ~= {per_round_mnist * 200 / 60:.1f} min "
          f"(~{per_round_mnist * 200 / 3600:.2f} h)")
    print(f"CIFAR-10: ~{per_round_cifar:.1f}s/round "
          f"-> 100 rounds ~= {per_round_cifar * 100 / 60:.1f} min "
          f"(~{per_round_cifar * 100 / 3600:.2f} h)")
    print("\nMultiply the per-run time above by (number of configs x number "
          "of seeds) to size a real compute budget.")
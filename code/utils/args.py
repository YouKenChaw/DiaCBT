import argparse


def setup_args():
    parser = argparse.ArgumentParser()
    # project config
    parser.add_argument("--random_seed", type=int, default=99)
    parser.add_argument("--do_train", type=str2bool, default=False)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="C2D2", choices=["C2D2", "PatternReframe"])
    parser.add_argument("--num_training_epochs", type=int, default=2)
    parser.add_argument("--sample_times", type=int, default=300)
    parser.add_argument("--max_turn", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--backbone", type=str, default="llama3-8b-instruct")
    parser.add_argument("--npc_backbone", type=str, default="llama3-8b-instruct")
    parser.add_argument("--reward_backbone", type=str, default="llama3-8b-instruct")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--debug", type=str2bool, default=True)
    parser.add_argument("--eval_steps", type=int, default=50)
    return parser.parse_args()


def print_args(args):
    print("-------------------- Arguments --------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append(f" {arg} {dots} {getattr(args, arg)}")
    for arg_str in sorted(str_list, key=lambda x: x.lower()):
        print(arg_str, flush=True)
    print("----------------- End of Arguments ----------------", flush=True)


def str2bool(v):
    return str(v).lower() in ("yes", "true", "1")


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

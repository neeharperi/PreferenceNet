import torch
from argparse import ArgumentParser
import numpy as np
import sys

"""builds n binary feature vectors for f features according to inputted probability distributions. features can either be 
mutually exclusive ([1,0], [0,1]), non-mutually exclusive ([1,1],[1,0],[0,0]), or both ! go crazy. args below."""


def features(prob, length):  # this is where non-mutually exclusive features are made
    feats = np.array([])
    for p in prob:
        one = np.ones(int(round(length * p)))  # frequency of feature option p occuring
        zero = np.zeros(int(length * (round(1 - p, 1))))  # frequency feature option p won't occur
        feats_col = np.concatenate((one, zero))
        np.random.shuffle(feats_col)  # randomize frequencies between vectors
        if len(feats_col) < length:  # sometimes not all features are accounted for cuz of small batch size
            feats_col = np.concatenate((feats_col, np.zeros(length - len(feats_col))))
        feats = np.concatenate((feats, feats_col[:length]))  # concat all e features
    return np.reshape(feats, (-1, len(prob)), order='F')


def me_features(prob, length): # this is where mutually exclusive features are birthed
    n = len(prob)
    prob = np.round(length * prob)  # the frequency of each option
    if prob.sum() < length:  # sometimes not all features are accounted for because of small batch size
        prob[0] = prob[0] + (length - prob.sum())
    feats = np.array([])
    categories = np.eye(n)  # all options
    for i in range(n):
        instances = np.tile(categories[i], int(prob[i]))  # frequencies actualized
        feats = np.concatenate((feats, instances))
    feats = feats.reshape(-1, n)[:length]
    np.random.shuffle(feats)                               # 1 vector arrays are always the same btw
    return feats


def load_categories(afeats):  # categorizes by ad preferences. same ad pref vectors = same category
    # returns tensor of length n, each value coinciding with n's category
    return [np.unique(afeats, return_inverse=True, axis=0)[1]]


# you need to make two files (1 representing ad preferences, and 1 of user qualities)
def generate_distance(ad_feats, u_feats):
    """  """
    n, m = ad_feats.shape[0], u_feats.shape[0]
    D = torch.zeros(n, m, m)  # distances per advertiser between every user
    for a in range(n):
        divisor = list(ad_feats[a]).count(1)  # counts how many features the advertiser cares about
        if divisor == 0:
            D[a] = torch.zeros(m, m)  # prevents ambivalent advertisers from breaking my code
            continue
        for t in range(m):
            for h in range(m):  # the ' * a_feats ' is the ad preference contribution to the distance func
                if t != h:
                    D[a][t][h] = sum(np.bitwise_xor(u_feats[t], u_feats[h]) * ad_feats[a]) / divisor
    return D if len(ad_feats) == 1 else D*2


def single_category(n):
    return [[i for i in range(n)]]


def uniform_distance(c, m, d):
    return torch.ones(c, m, m)*d


# Let's fix at 3x4
def experiment_1():
    ad_feats = torch.tensor([[0, 0, 1]])
    u_feats = torch.tensor([
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ])
    return generate_distance(ad_feats, u_feats)


def experiment_2():
    ad_feats = torch.tensor([[0, 0, 1]])
    u_feats = torch.tensor([
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0]
    ])
    return generate_distance(ad_feats, u_feats)


def experiment_3():
    ad_feats = torch.tensor([[0, 0, 1]])
    u_feats = torch.tensor([
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0]
    ])
    return generate_distance(ad_feats, u_feats)


parser = ArgumentParser()

# input probabilities as a tuple of floats, seperated by spaces.
parser.add_argument('--n', type=int, default=2)  # number of vectors
parser.add_argument('--p_me', type=float, nargs='*', action='append', default=[])  # Must add up to 1!
parser.add_argument('--p_feats', type=float, nargs='*', default=[])  # for all non-m.e features
parser.add_argument('--file_name', default='default')  # name of file where tensors are kept

# these args are just to test out the distance function separately and save it to a file if you'd like (use --file_name)
parser.add_argument('--batch', type=int, nargs='*', default=[])  # a tuple of dimensions (n m) n agents, m items
parser.add_argument('--ad_file', type=str, default="")  # name of the file for ad preferences
parser.add_argument('--user_file', type=str, default="")  # name of the file for user features

parser.add_argument('--experiment', type=int, nargs='*', default=[])  # convenience method for hardcoding experiment setups


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.file_name
    if args.experiment:
        if args.experiment[0] == 1:
            print(experiment_1(args.experiment[1]))
        exit()

    if args.batch == []:
        n = args.n
        if len(args.p_me) + len(args.p_feats) == 0:
            sys.exit("no features! try again dude")
        f = 0
        while f != len(args.p_me):
            p = np.array(args.p_me[f], dtype=float)
            if p.sum() == 1:
                f += 1
                if f == 1:
                    feats = me_features(p, n)
                else:
                    feats = np.concatenate((feats, me_features(p, n)), axis=1)
            else:
                print(f"feature {f + 1} does not add up to 1")
                sys.exit("try again dude")

        if len(args.p_feats) != 0:
            p = list(np.array(args.p_feats, dtype=float))
            if 'feats' in locals():
                feats = np.concatenate((feats, features(p, n)), axis=1)
            else:
                feats = features(p, n)


        print(f"\nfeatures:\n {feats}\nsaved in {filename}.pt\nhooray.")
    else:
        afeats = torch.load(args.ad_file).astype(int)
        ufeats = torch.load(args.user_file).astype(int)
        feats = generate_distance(args.batch, afeats, ufeats)
        print(feats)
    torch.save(feats, filename + ".pt")

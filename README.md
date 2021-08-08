# PreferenceNet: Encoding Human Preferences in Auction Design with Deep Learning [[PDF](https://arxiv.org/pdf/2106.03215.pdf)]
Neehar Peri*, Michael J. Curry*, Samuel Dooley, and John P. Dickerson

The design of optimal auctions is a problem of interest in economics, game theory and computer science. Despite decades of effort, strategyproof, revenue-maximizing auction designs are still not known outside of restricted settings. However, recent methods using deep learning have shown some success in approximating optimal auctions, recovering several known solutions and outperforming strong baselines when optimal auctions are not known. In addition to maximizing revenue, auction mechanisms may also seek to encourage socially desirable constraints such as allocation fairness or diversity. However, these philosophical notions neither have standardization nor do they have widely accepted formal definitions. In this paper, we propose PreferenceNet, an extension of existing neural-network-based auction mechanisms to encode constraints using (potentially human-provided) exemplars of desirable allocations. In addition, we introduce a new metric to evaluate an auction allocations' adherence to such socially desirable constraints and demonstrate that our proposed method is competitive with current state-of-the-art neural-network based auction designs. We validate our approach through human subject research and show that we are able to effectively capture real human preferences.

# Using this Code Base
**diversity/** - RegretNet + Entropy Loss
```
python train.py --diversity entropy 1.0  --dataset {n}x{m}-{pv|mv} 1 {--no_lagrange}
```

**fairness/** - RegretNet + TVF Loss
```
python train.py --fairness tvf 0.0  --dataset {n}x{m}-{pv|mv} 1 {--no_lagrange}
```

**quota/** - RegretNet + Quota Loss
```
python train.py --quota quota 1.0  --dataset {n}x{m}-{pv|mv} 1 {--no_lagrange}
```

**learnable_preferences/** - RegretNet + PreferenceLoss
```
python train.py --preference entropy_ranking {a} tvf_ranking {b} quota_quota {c} human_preference {d} \
--dataset {n}x{m}-{pv|mv} 1 {--preference-file ../path/to/preference/file.pth} {--preference-label-noise 1.0} {--lagrange}
```

**compare_mechanism/** - Evaluation Code
```
python evaluate.py --preference entropy_ranking {a} tvf_ranking {b} quota_quota {c} human_preference {d} \ 
--dataset {n}x{m}-{pv|mv} 1 {--preference-file ../path/to/preference/file.pth} {--preference-label-noise 1.0} {--lagrange}
```

**survey/** - Survey Data and Analysis

**Parameters**

{n} - Number of Agents

{m} - Number of Items

{pv|mv} - Pavlov Auction or Manelli-Vincent Auction

{--no_lagrange} - Train RegretNet + Constraint without Lagrange Multipliers

{a} {b} {c} {d} - Floating point values such that a + b + c + d = 1.0

{--preference-file ../path/to/preference/file.pth} - Training Labels from Human Survey (Only Necessary if d > 0)

{--preference-label-noise 1.0} - Add Training Noise according to Probit Model

{--lagrange} - Train PreferenceNet with Lagrangian Multipliers

# Acknowledgements
This research was supported in part by NSF CAREER Award IIS-1846237, NSF D-ISN Award #2039862, NSF Award CCF-1852352, NIH R01 Award NLM-013039-01, NIST MSE Award #20126334, DARPA GARD #HR00112020007, DoD WHS Award #HQ003420F0035, and a Google Faculty Research Award. The code for the original RegretNet, in TensorFlow, is available [here](https://github.com/saisrivatsan/deep-opt-auctions). We thank the authors of ProportionNet for sharing their codebase and Kevin Kuo and Uro Lyi for their feedback in writing this paper.

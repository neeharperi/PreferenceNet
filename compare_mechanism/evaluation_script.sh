######################
python evaluate.py --model-path ../learnable_preferences/result/entropy_ranking_1.0/1x2-pv_1_synthetic_1.0_noise_0_0/98fe36850383af98750e6daa910d32da/best_checkpoint.pt --preference entropy_threshold 0.5 tvf_threshold 0.5 --dataset 1x2-pv 1
python evaluate.py --model-path ../learnable_preferences/result/tvf_ranking_1.0/1x2-pv_1_synthetic_1.0_noise_0_0/5e8ddda6615c1700eb549a64de8b2aa8/best_checkpoint.pt --preference entropy_threshold 0.5 tvf_threshold 0.5 --dataset 1x2-pv 1
python evaluate.py --model-path ../learnable_preferences/result/entropy_ranking_1.0/1x2-mv_1_synthetic_1.0_noise_0_0/fcc7edaf12174531d2b4c7e3c4f0035b/best_checkpoint.pt --preference entropy_threshold 0.5 tvf_threshold 0.5 --dataset 1x2-mv 1
python evaluate.py --model-path ../learnable_preferences/result/tvf_ranking_1.0/1x2-mv_1_synthetic_1.0_noise_0_0/f35ea65feab83c6f075683fbc9586be0/best_checkpoint.pt --preference entropy_threshold 0.5 tvf_threshold 0.5 --dataset 1x2-mv 1
######################
python distance_matrix.py
mkdir files_hotpotqa_t/
mkdir files_hotpotqa_v/
mkdir files_hotpotqa_ts_t/
mkdir files_hotpotqa_ts_v/

pip install -U -r requirements.txt

# LinUCB
python analysis_shortanswer.py hotpotqa
python train_shortanswer.py hotpotqa
python analysis_shortanswer_patched.py hotpotqa

python report_metrics.py hotpotqa
python report_metrics_patched.py hotpotqa

# Thompson Sampling
python analysis_shortanswer.py hotpotqa_ts
python train_shortanswer.py hotpotqa_ts
python analysis_shortanswer_patched.py hotpotqa_ts

python report_metrics.py hotpotqa_ts
python report_metrics_patched.py hotpotqa_ts

# Baselines
python report_metrics_patched.py hotpotqa_paraph
python report_metrics_patched.py hotpotqa_top20
python report_metrics_patched.py hotpotqa_bestarm

python train_shortanswer.py hotpotqa_nogate
python analysis_shortanswer_patched.py hotpotqa_nogate
python report_metrics_patched.py hotpotqa_nogate

python train_shortanswer.py hotpotqa_nocost
python analysis_shortanswer_patched.py hotpotqa_nocost
python report_metrics_patched.py hotpotqa_nocost

python analysis_shortanswer_posthoc.py hotpotqa
python report_metrics_patched.py hotpotqa_posthoc

python train_shortanswer.py hotpotqa_tb
python analysis_shortanswer_patched.py hotpotqa_tb
python report_metrics_patched.py hotpotqa_tb

python train_shortanswer.py hotpotqa_lb
python analysis_shortanswer_patched.py hotpotqa_lb
python report_metrics_patched.py hotpotqa_lb
mkdir files_fever_t/
mkdir files_fever_v/
mkdir files_fever_ts_t/
mkdir files_fever_ts_v/

pip install -U -r requirements.txt

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip

python create_knowledge_base.py fever

# LinUCB
python analysis.py fever
python train.py fever
python analysis_patched.py fever

python report_metrics.py fever
python report_metrics_patched.py fever

# Thompson Sampling
python analysis.py fever_ts
python train.py fever_ts
python analysis_patched.py fever_ts

python report_metrics.py fever_ts
python report_metrics_patched.py fever_ts

# Baselines
python report_metrics_patched.py fever_paraph
python report_metrics_patched.py fever_top20
python report_metrics_patched.py fever_bestarm

python train.py fever_nogate
python analysis_patched.py fever_nogate
python report_metrics_patched.py fever_nogate

python train.py fever_nocost
python analysis_patched.py fever_nocost
python report_metrics_patched.py fever_nocost

python analysis_posthoc.py fever
python report_metrics_patched.py fever_posthoc

python train.py fever_tb
python analysis_patched.py fever_tb
python report_metrics_patched.py fever_tb

python train.py fever_lb
python analysis_patched.py fever_lb
python report_metrics_patched.py fever_lb
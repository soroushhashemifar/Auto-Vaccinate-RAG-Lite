setup.sh

python create_knowledge_base.py

python main.py
python evaluation.py

python train_patcher.py
python main_with_patch.py
python report_metrics_prepatch.py
python report_metrics_postpatch.py
python report_deltas.py
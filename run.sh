export PYTHONPATH=$PYTHONPATH:.

mkdir files_fever_t/
mkdir files_fever_v/
mkdir files_fever_ts_t/
mkdir files_fever_ts_v/
mkdir files_hotpotqa_t/
mkdir files_hotpotqa_v/
mkdir files_hotpotqa_ts_t/
mkdir files_hotpotqa_ts_v/

pip install -U -r requirements.txt

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip
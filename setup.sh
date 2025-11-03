mkdir out/

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip

wget https://thespermwhale.com/jaseweston/babi/movieqa.tar.gz
tar -zxvf /content/movieqa.tar.gz

pip install -r requirements.txt
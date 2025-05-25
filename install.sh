git clone https://github.com/openai/human-eval
pip install -e human-eval
pip install openai antrhopic nltk numpy datasets
apt install jq netcat-openbsd -y

# python -m simple-evals.simple_evals --model grader_ablation_3b
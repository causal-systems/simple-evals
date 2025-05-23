git clone https://github.com/openai/human-eval
pip install -e human-eval
pip install openai
pip install anthropic
apt install jq netcat-openbsd -y

# python -m simple-evals.simple_evals --model grader_ablation_3b
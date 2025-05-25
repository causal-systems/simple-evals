git clone https://github.com/openai/human-eval
pip install -e human-eval
git clone https://github.com/hazyresearch/legalbench
pip install openai anthropic nltk numpy datasets scikit-learn
apt install jq netcat-openbsd -y

# python -m simple-evals.simple_evals --model grader_ablation_3b
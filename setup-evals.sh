cd /root
apt update
apt install tmux vim unzip -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
git clone git@github.com:causal-systems/sft
git clone git@github.com:causal-systems/simple-evals
./sft/install.sh
cd /root
pip install boto3 anthropic openai tabulate
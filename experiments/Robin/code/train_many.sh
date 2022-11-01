for i in "koelectra_petition_only_config" "koelectra_nsmc_only_config" "koelectra_slack_only_config"
do
    python3 train.py \
        --config $i
done
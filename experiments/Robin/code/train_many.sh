for i in $("config/kcelectra_petition_only_config.yaml" "config/kcelectra_nsmc_only_config" "config/kcelectra_slack_only_config")
do
    python3 train.py \
        --config $i \

done
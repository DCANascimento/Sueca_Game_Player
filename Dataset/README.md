# IMPORTANT INFORMATION

This was taken straight from another project where two of this group's members participate, cutting some of the unnecessary parts (but not all of them). If the goal is to run another simulation please create a venv with folder name of ".venv" and install all the needed libraries in said venv (present in the requirements.txt file).

When it comes to actually running the data_gatherer itself, then please utilize the following command, as the data gathering script has a lot of optional arguments and it could cause confusion. The command should be run in the "data-gathering/" directory and is as follows:

    SUECA_STATISTICS_FAST_MODE=1 SUECA_MQTT_EVENTS=false SUECA_BOT_THINK_TIME=0   ./.venv/bin/python statistic-analysis/data_gatherer.py     --matches 10000     --fast-inproc     --split-csv     --no-game-files     --output-dir statistic-analysis/batch_output_10000     --poll-interval 0.0 --combinations-file ./statistic-analysis/combinations.example.json

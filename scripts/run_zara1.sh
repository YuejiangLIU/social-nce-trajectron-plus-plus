#!/usr/bin/env bash

DATASET=zara1
weight_grid=(0.0 0.5 1.0 2.0 3.0 5.0)

for w in ${weight_grid[@]}; do
	echo 'WEIGHT:' ${w}
	bash run_train.sh ${DATASET} ${w} &		# remove '&'' for serial jobs
done

wait

for w in ${weight_grid[@]}; do
	echo 'WEIGHT:' ${w}
	bash run_eval.sh ${DATASET} ${w} &
done

#!/usr/bin/env bash

DATASET=univ
weight_grid=(0.0 10.0 20.0 30.0 50.0 70.0)

for w in ${weight_grid[@]}; do
	echo 'WEIGHT:' ${w}
	bash run_train.sh ${DATASET} ${w} &		# remove '&'' for serial jobs
done

wait

for w in ${weight_grid[@]}; do
	echo 'WEIGHT:' ${w}
	bash run_eval.sh ${DATASET} ${w} &
done

#!/bin/bash

OUTDIR=$1
NUM_REDUCERS=8

hdfs dfs -rm -r -skipTrash $OUTDIR*

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -files mapper.py,reducer.py \
    -numReduceTasks ${NUM_REDUCERS} \
    -mapper "python3.6 mapper.py" \
    -combiner "python3.6 reducer.py" \
    -reducer "python3.6 reducer.py" \
    -input hdfs:///data/wiki/en_articles_part \
    -output $OUTDIR

for (( i = 0; i < NUM_REDUCERS; ++i ))
do
    hdfs dfs -cat $OUTDIR/part-0000${i} | head
done
#!/bin/bash

OUTDIR=$1
NUM_REDUCERS=8

MID_DIR=$1-mid

hdfs dfs -rm -r -skipTrash $OUTDIR*

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -files mapper.py,reducer.py,/datasets/stop_words_en.txt \
    -numReduceTasks ${NUM_REDUCERS} \
    -mapper "python3.6 mapper.py" \
    -combiner "python3.6 reducer.py" \
    -reducer "python3.6 reducer.py" \
    -input hdfs:///data/wiki/en_articles \
    -output $MID_DIR

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
    -D mapreduce.partition.keycomparator.options="-k1nr" \
    -files inverter.sh \
    -numReduceTasks 1 \
    -mapper "bash inverter.sh" \
    -reducer "bash inverter.sh" \
    -input $MID_DIR \
    -output $OUTDIR

for (( i = 0; i < 1; ++i ))
do
    hdfs dfs -cat $OUTDIR/part-0000${i} | head
done

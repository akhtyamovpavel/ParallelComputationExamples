#!/bin/bash

cat | awk 'BEGIN {OFS="\t"} {print $2, $1}'

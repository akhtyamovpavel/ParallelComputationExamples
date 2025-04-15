-- EXTERNAL - view over data
CREATE EXTERNAL TABLE
Subnets 
( ip STRING, mask STRING )
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '/data/subnets/variant1';

SELECT ip FROM subnets LIMIT 10;

SELECT COUNT(DISTINCT mask) FROM subnets;

SELECT mask, COUNT(ip)
FROM subnets
GROUP BY mask;

CREATE TABLE SubnetsPartitioned (
    ip STRING
)
PARTITIONED BY (mask STRING)
STORED AS TEXTFILE;

SET hive.exec.dynamic.partition.mode=nonstrict;
INSERT OVERWRITE TABLE SubnetsPartitioned PARTITION (mask)
SELECT ip, mask FROM Subnets;

SELECT * FROM subnetspartitioned WHERE mask = '255.255.255.255.0';

SELECT mask, COUNT(ip)
FROM subnetspartitioned
GROUP BY mask;

DROP table logsSerde;

CREATE EXTERNAL TABLE
logsSerDe
(
    ip STRING,
    dom STRING,
    mon STRING,
    `year` STRING,
    ho STRING,
    mi STRING,
    sec STRING,
    method STRING,
    path STRING,
    protocol STRING,
    return_code STRING,
    referer STRING,
    user_agent STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\d+\\.\\d+\\.\\d+\\.\\d+)\\s+-\\s+-\\s+\\[(\\d+)/(\\w+)/(\\d+):(\\d+):(\\d+):(\\d+) .*\\]\\s+\\"(\\w+)\\s+([\\/\\-\\w\\.]+)\\s+([\\/\\w\\.]+)\\"\\s+(\\d+)\\s+\\d+\\s+\\"([\\/\\-\\w\\.]+)\\"\\s+\\"([^\\"]+)\\"$'
)
LOCATION '/data/access_logs/big_log';

SELECT * FROM logsserde LIMIT 10;

add jar /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;
SELECT dom, COUNT(return_code)
FROM logsserde
GROUP by dom;

package ru.pd.udfs;

import org.apache.hadoop.hive.ql.exec.UDF;

public class IdentityUDF extends UDF {
    public String evaluate(String str) {
        return str;
    }
}

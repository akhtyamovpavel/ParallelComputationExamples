package ru.pd.udfs;

import org.apache.hadoop.hive.ql.exec.UDF;

public class SumOctetsUdf extends UDF {

    public int evaluate(String string) {
        int result = 0;
        for (String token : string.split("\\.")) {
            result += Integer.parseInt(token);
        }
        return result;
    }
}

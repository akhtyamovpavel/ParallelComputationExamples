package ru.pd.udfs;

import static junit.framework.Assert.assertEquals;

public class TestIdentityUDF {
    public void testUdf() {
        IdentityUDF udf = new IdentityUDF();
        assertEquals("hello", udf.evaluate("hello"));
    }
}

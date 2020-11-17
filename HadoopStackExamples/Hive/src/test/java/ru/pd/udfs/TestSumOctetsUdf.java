package ru.pd.udfs;

import org.junit.jupiter.api.Test;

import static junit.framework.Assert.assertEquals;

public class TestSumOctetsUdf {

    @Test
    public void testSumOctets() {
        SumOctetsUdf udf = new SumOctetsUdf();
        assertEquals(128, udf.evaluate("32.32.26.38"));
    }
}

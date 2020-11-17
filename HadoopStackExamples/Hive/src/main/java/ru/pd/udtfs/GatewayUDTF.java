package ru.pd.udtfs;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;

import java.util.ArrayList;

public class GatewayUDTF extends GenericUDTF {
    private StringObjectInspector ipInspector;
    private StringObjectInspector maskInspector;

    private Object forwardArray[] = new Object[3];

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException(getClass().getSimpleName() + "takes only two arguments");
        }

        ipInspector = (StringObjectInspector) argOIs[0];
        maskInspector = (StringObjectInspector) argOIs[1];

        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldsOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("ip");
        fieldNames.add("gateway");
        fieldNames.add("mask");
        fieldsOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldsOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldsOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldsOIs);
    }

    public void process(Object[] objects) throws HiveException {
        String ip = ipInspector.getPrimitiveJavaObject(objects[0]);
        String mask = maskInspector.getPrimitiveJavaObject(objects[1]);

        String[] ipTokens = ip.split("\\.");
        String[] maskTokens = mask.split("\\.");

        StringBuilder gatewayBuilder = new StringBuilder();
        for (int index = 0; index < 4; ++index) {
            if (index > 0) {
                gatewayBuilder.append('.');
            }
            gatewayBuilder.append(
                Integer.parseInt(ipTokens[index]) & Integer.parseInt(maskTokens[index])
            );
        }
        String gateway = gatewayBuilder.toString();

        forwardArray[0] = ip;
        forwardArray[1] = gateway;
        forwardArray[2] = mask;
        forward(forwardArray);
    }

    public void close() throws HiveException {

    }
}

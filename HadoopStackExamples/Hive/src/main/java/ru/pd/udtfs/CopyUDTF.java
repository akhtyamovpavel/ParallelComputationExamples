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

public class CopyUDTF extends GenericUDTF {
    private StringObjectInspector inspector;

    private Object forwardArray[] = new Object[1];

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException(getClass().getSimpleName() + "takes only one argument");
        }

        inspector = (StringObjectInspector) argOIs[0];
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldsOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("Ips");
        fieldsOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldsOIs);

    }

    public void process(Object[] objects) throws HiveException {
        String address = inspector.getPrimitiveJavaObject(objects[0]);
        forwardArray[0] = address;
        forward(forwardArray);
        forward(forwardArray);
    }

    public void close() throws HiveException {

    }
}

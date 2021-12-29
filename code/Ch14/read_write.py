#!/usr/bin/env python3

# tag::ch14-params-read-write[]

from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class ScalarNAFiller(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasInputCols,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    # ... rest of the class here

class _ExtremeValueCapperParams(
    HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable
):
    # ... rest of the class here

# end::ch14-params-read-write[]

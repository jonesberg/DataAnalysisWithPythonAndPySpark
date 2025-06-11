# Databricks notebook source
# MAGIC %md
# MAGIC # Getting ready for the book's examples using Databricks Free.
# MAGIC
# MAGIC If you want to get ready ASAP, you can run this notebook end-to-end. The data will be available in volumes and tables under `dapp`. I recommend for you to read carefully through the notebook and use the `Catalog` link aside to understand what to change when reading from a Databricks volume/table.
# MAGIC
# MAGIC ## Summary of changes
# MAGIC
# MAGIC You don't have to set the `spark` variable.
# MAGIC
# MAGIC # Chapter 2 and 3 (provided as an example)
# MAGIC
# MAGIC **Listing 2.1:** Not applicable using Databricks Free
# MAGIC
# MAGIC **Listing 2.2:** Not applicable using Databricks Free
# MAGIC
# MAGIC **Listing 2.3:** Will not work using Databricks Free / Spark Serverless. You can safely ignore
# MAGIC
# MAGIC **Listing 2.5:** Use the location of the volume ins of the (local) location provided in the booktead: `spark.read.text("/Volumes/dapp/gutenberg/data/1342-0.txt")`
# MAGIC
# MAGIC **Listing 2.9:** You can also use `display()` in lieu of `show()`.
# MAGIC
# MAGIC **Listings 3.3 and 3.4:** Change the location of the writing to a Databricks Volume `/Volumes/dapp/gutenberg/data/simple_count.csv`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create a catalog to store our data
# MAGIC create catalog if not exists `dapp`;
# MAGIC
# MAGIC -- With the catalog created, let's create the schemas
# MAGIC use catalog `dapp`;
# MAGIC
# MAGIC create schema if not exists `broadcast_logs`;
# MAGIC
# MAGIC create schema if not exists `elements`;
# MAGIC
# MAGIC create schema if not exists `gsod_noaa`;
# MAGIC
# MAGIC create schema if not exists `gutenberg`;
# MAGIC
# MAGIC create schema if not exists `list_of_numbers`;
# MAGIC
# MAGIC create schema if not exists `recipes`;
# MAGIC
# MAGIC create schema if not exists `shows`;
# MAGIC
# MAGIC create schema if not exists `window`;

# COMMAND ----------

# MAGIC %sql
# MAGIC create volume if not exists broadcast_logs.data;
# MAGIC create volume if not exists broadcast_logs.ReferenceTables;
# MAGIC create volume if not exists elements.data;
# MAGIC create volume if not exists gsod_noaa.data;
# MAGIC create volume if not exists gutenberg.data;
# MAGIC create volume if not exists list_of_numbers.data;
# MAGIC create volume if not exists recipes.data;
# MAGIC create volume if not exists shows.data;
# MAGIC create volume if not exists window.gsod_light;
# MAGIC create volume if not exists window.gsod;

# COMMAND ----------

# MAGIC %sh
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/BroadcastLogs_2018_Q3_M8_sample.CSV" --output /Volumes/dapp/broadcast_logs/data/BroadcastLogs_2018_Q3_M8_sample.CSV
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/Call_Signs.csv" --output /Volumes/dapp/broadcast_logs/data/Call_Signs.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/BroadcastProducers.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/BroadcastProducers.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_AirLanguage.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_AirLanguage.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_AudienceTargetAge.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_AudienceTargetAge.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_AudienceTargetEthnic.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_AudienceTargetEthnic.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_BroadcastOriginPoint.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_BroadcastOriginPoint.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_Category.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_Category.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_ClosedCaption.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_ClosedCaption.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_Composition.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_Composition.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_CountryOfOrigin.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_CountryOfOrigin.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_DubDramaCredit.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_DubDramaCredit.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_EthnicProgram.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_EthnicProgram.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_Exhibition.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_Exhibition.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_FilmClassification.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_FilmClassification.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_NetworkAffiliation.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_NetworkAffiliation.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_ProductionSource.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_ProductionSource.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_ProgramClass.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_ProgramClass.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/CD_SpecialAttention.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/CD_SpecialAttention.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/broadcast_logs/ReferenceTables/LogIdentifier.csv" --output /Volumes/dapp/broadcast_logs/ReferenceTables/LogIdentifier.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/elements/Periodic_Table_Of_Elements.csv" --output /Volumes/dapp/elements/data/Periodic_Table_Of_Elements.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2010.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2010.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2011.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2011.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2012.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2012.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2013.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2013.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2014.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2014.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2015.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2015.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2016.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2016.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2017.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2017.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2018.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2018.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2019.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2019.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gsod_noaa/gsod2020.parquet" --output /Volumes/dapp/gsod_noaa/data/gsod2020.parquet

# COMMAND ----------

# MAGIC %sh
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/11-0.txt" --output /Volumes/dapp/gutenberg/data/11-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/1342-0.txt" --output /Volumes/dapp/gutenberg/data/1342-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/1661-0.txt" --output /Volumes/dapp/gutenberg/data/1661-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/2701-0.txt" --output /Volumes/dapp/gutenberg/data/2701-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/30254-0.txt" --output /Volumes/dapp/gutenberg/data/30254-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/gutenberg_books/84-0.txt" --output /Volumes/dapp/gutenberg/data/84-0.txt
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/list_of_numbers/sample.csv" --output /Volumes/dapp/list_of_numbers/data/sample.csv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/recipes/epi_r.csv" --output /Volumes/dapp/recipes/data/epi_r.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/window/gsod_light.parquet/part-00000-49044fc1-973b-4ca6-a071-621e170220a9-c000.snappy.parquet" --output /Volumes/dapp/window/gsod_light/part-00000-49044fc1-973b-4ca6-a071-621e170220a9-c000.snappy.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/window/gsod.parquet/part-00000-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet" --output /Volumes/dapp/window/gsod/part-00000-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/window/gsod.parquet/part-00001-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet" --output /Volumes/dapp/window/gsod/part-00001-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/window/gsod.parquet/part-00002-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet" --output /Volumes/dapp/window/gsod/part-00002-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/window/gsod.parquet/part-00003-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet" --output /Volumes/dapp/window/gsod/part-00003-6ccefe44-8397-4020-b02f-832dd89c20a6-c000.snappy.parquet
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/shows/pokedex.dsv" --output /Volumes/dapp/shows/data/pokedev.dsv
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/shows/shows-breaking-bad.json" --output /Volumes/dapp/shows/data/shows-breaking-bad.json
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/shows/shows-golden-girls.json" --output /Volumes/dapp/shows/data/shows-golden-girls.json
# MAGIC curl "https://raw.githubusercontent.com/jonesberg/DataAnalysisWithPythonAndPySpark-Data/refs/heads/trunk/shows/shows-silicon-valley.json" --output /Volumes/dapp/shows/data/shows-silicon-valley.json

# COMMAND ----------

spark.read.parquet("/Volumes/dapp/window/gsod_light/*.parquet").write.mode("overwrite").saveAsTable("dapp.window.gsod_light")
spark.read.parquet("/Volumes/dapp/window/gsod/*.parquet").write.mode("overwrite").saveAsTable("dapp.window.gsod")

# COMMAND ----------


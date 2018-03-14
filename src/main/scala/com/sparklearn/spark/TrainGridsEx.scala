package com.sparklearn.spark

import com.sparklearn.spark.LR.getClass
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object TrainGridsEx {
  def main(args: Array[String]): Unit = {

    import org.apache.spark.SparkConf

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    // READ data
    val path = getClass.getResource("/USA_Housing.csv").getPath
    val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(path)

    data.printSchema()

    val df = data.select(data("Price").as("label"),
      $"Avg Area Income",
      $"Avg Area House Age",
      $"Avg Area Number of Rooms",
      $"Avg Area Number of Bedrooms",
      $"Area Population",
      $"Price")

    df.printSchema

    val assembler = new VectorAssembler().setInputCols(Array(
      "Avg Area Income",
      "Avg Area House Age",
      "Avg Area Number of Rooms",
      "Avg Area Number of Bedrooms",
      "Area Population",
      "Price"))
  }
}

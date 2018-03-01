package com.sparklearn.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

object LR {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.SparkConf

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    val path = getClass.getResource("/USA_Housing.csv").getPath
    val data = spark.read.option("header", "true").option("inferSchema", value = true).format("csv").load(path)

    printData(data)

    import spark.implicits._

    // ("label", "features")
    val df = data.select(data("Price").as("label"),
      $"Avg Area Income",
      $"Avg Area House Age",
      $"Avg Area Number of Rooms",
      $"Avg Area Number of Bedrooms",
      $"Area Population",
      $"Price")

    val assembler = new VectorAssembler().setInputCols(Array(
      "Avg Area Income",
      "Avg Area House Age",
      "Avg Area Number of Rooms",
      "Avg Area Number of Bedrooms",
      "Area Population",
      "Price")).setOutputCol("features")

    val output = assembler.transform(df).select($"label", $"features")

    val lr = new LinearRegression()
    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary
    trainingSummary.residuals.show()
    trainingSummary.predictions.show()
    println(s"r2: ${trainingSummary.r2}")
    println(s"r2: ${trainingSummary.rootMeanSquaredError}")

    spark.close()
  }

  private def printData(data: DataFrame): Unit = {
    val colNames = data.columns
    val firstRow = data.head(1)(0)
    println("\n")
    println("Example data row: ")
    for (ind <- Range(1, colNames.length)) {
      println(colNames(ind))
      println(firstRow(ind))
      println("\n")
    }
  }
}

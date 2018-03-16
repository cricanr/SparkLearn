package com.sparklearn.spark

import com.sparklearn.spark.LR.getClass
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
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

    import spark.implicits._

    val df = data.select(
      data("Price").as("label"),
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
      "Area Population",
      "Price")).setOutputCol("features")

    val output = assembler.transform(df).select($"label", $"features")

    val Array(training, test) = output.select("label", "features").randomSplit(Array(0.7, 0.3), seed = 12345)

    // model
    val lr = new LinearRegression()

    val doubleParam: DoubleParam = lr.regParam
    //  FIXME: val paramGrid = new ParamGridBuilder().addGrid(doubleParam, Array(10000, 0,1)).build()
    val paramGrid = new ParamGridBuilder().build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    println(s"trainValidationSplit: $trainValidationSplit")

    val model = trainValidationSplit.fit(training)

    println(s"model.validationMetrics: ${model.validationMetrics.foreach(metric => println(s"$metric"))}")

    model.transform(test).select("features", "label", "prediction").show()
  }
}

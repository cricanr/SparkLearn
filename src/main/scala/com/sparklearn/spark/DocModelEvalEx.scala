package com.sparklearn.spark

import com.sparklearn.spark.LogRegExample.getClass
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object DocModelEvalEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    val path = getClass.getResource("/sample_linear_regression_data.txt").getPath
    val data = spark.read.format("libsvm").load(path)

    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    data.printSchema()

    val lr = new LinearRegression()
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()

    println(s"ParamGrid: ${paramGrid(0)}")

    val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator()).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

    val model = trainValidationSplit.fit(training)

    model.transform(test).select("features", "label", "prediction").show()

    println(s"Best model: ${model.bestModel}")
  }
}
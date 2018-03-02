package com.sparklearn.spark

import com.sparklearn.spark.Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LogRegExample {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.SparkConf

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    println("Run Logistic regression on Titanic data and print out statistics:")
    logRegTitanic(spark)

    println("-----------------------------------------------------------")

    println("Run advertising logistic regression and print out statistics:")
    logRegAdvertising(spark)

    spark.close()
  }

  private def logRegTitanic(spark: SparkSession) = {
    val path = getClass.getResource("/titanic.csv").getPath

    val data = spark.read.option("header", "true").option("inferSchema", value = true).format("csv").load(path)

    data.printSchema()

    printData(data)

    import spark.implicits._

    val logRegDataAll = data.select(data("Survived").as("label"),
      $"Pclass",
      $"Name",
      $"Sex",
      $"Age",
      $"SibSp",
      $"Parch",
      $"Fare",
      $"Embarked")

    val logRegData = logRegDataAll.na.drop()

    // Converting Strings to numerical values
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    // Convert numerical values into One Hot Encoding (0 or 1)
    val genderEncoder = new OneHotEncoderEstimator().setInputCols(Array("SexIndex")).setOutputCols(Array("SexVec"))
    val embarkEncoder = new OneHotEncoderEstimator().setInputCols(Array("EmbarkedIndex")).setOutputCols(Array("EmbarkVec"))

    // ("label", "features")
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec"))
      .setOutputCol("features")

    val Array(training, test) = logRegData.randomSplit(Array(0.7, 0.3), seed = 12345)

    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

    val model = pipeline.fit(training)

    val results = model.transform(test)

    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

    println(s"predictionAndLabels: $predictionAndLabels")

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix: ")
    println(metrics.confusionMatrix)
  }

  private def logRegAdvertising(spark: SparkSession) = {
    val path = getClass.getResource("/advertising.csv").getPath

    val data = spark.read.option("header", "true").option("inferSchema", value = true).format("csv").load(path)

    data.printSchema()

    printData(data)

    import spark.implicits._

    val logRegDataAll = data.select(data("Clicked on Ad").as("label"),
      $"Daily Time Spent on Site",
      $"Age",
      $"Area Income",
      $"Daily Internet Usage",
      $"Male",
      $"Timestamp")

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage"))
      .setOutputCol("features")

    val Array(training, test) = logRegDataAll.randomSplit(Array(0.7, 0.3), 12345)

    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val model = pipeline.fit(training)

    val results = model.transform(test)

    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix: ")
    println(metrics.confusionMatrix)
  }
}
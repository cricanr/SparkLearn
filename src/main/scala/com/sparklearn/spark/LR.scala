package com.sparklearn.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import Utils._

object LR {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.SparkConf

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    println("Run USA Housing linear regression and print out statistics:")
    housingLinReg(spark)

    println("-----------------------------------------------------------")

    println("Run Ecommerce linear regression and print out statistics:")
    eCommerceLinReg(spark)

    spark.close()
  }

  private def eCommerceLinReg(spark: SparkSession) = {
    val path = getClass.getResource("/Ecommerce.csv").getPath

    val data = spark.read.option("header", "true").option("inferSchema", value = true).format("csv").load(path)

    data.printSchema()

    printData(data)

    import spark.implicits._

    // ("label", "features")

    // Rename the Yearly Amount Spent Column as "label"
    // Also grab only the numerical columns from the data
    // Set all of this as a new dataframe called df

    val df = data.select(data("Yearly Amount Spent").as("label"),
      $"Avg Session Length",
      $"Time on App",
      $"Time on Website",
      $"Length of Membership",
      $"Yearly Amount Spent")

    // An assembler converts the input values to a vector
    // A vector is what the ML algorithm reads to train a model

    // Use VectorAssembler to convert the input columns of df
    // to a single output column of an array called "features"
    // Set the input columns from which we are supposed to read the values.
    // Call this new object assembler

    val assembler = new VectorAssembler().setInputCols(Array(
      "Avg Session Length",
      "Time on App",
      "Time on Website",
      "Length of Membership",
      "Yearly Amount Spent"
    )).setOutputCol("features")

    // Use the assembler to transform our DataFrame to the two columns: label and features
    val output = assembler.transform(df).select($"label", $"features")

    // Create a Linear Regression Model object
    // Fit the model to the data and call this model lrModel
    val linearRegression = new LinearRegression()
    val lrModel = linearRegression.fit(output)

    // Print the coefficients and intercept for linear regression
    println(s"LrModel coefficients: ${lrModel.coefficients}")
    println(s"LrModel intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics!
    // Use the .summary method off your model to create an object
    // called trainingSummary
    val trainingSummary = lrModel.summary
    trainingSummary.residuals.show(20)

    // Show the residuals, the RMSE, the MSE, and the R^2 Values.
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"R^2: ${trainingSummary.r2}")
  }

  private def housingLinReg(spark: SparkSession): Unit = {
    val path = getClass.getResource("/USA_Housing.csv").getPath
    val data = spark.read.option("header", "true").option("inferSchema", value = true).format("csv").load(path)

    data.printSchema()

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
  }
}

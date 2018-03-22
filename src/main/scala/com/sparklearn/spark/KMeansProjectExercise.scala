
/////////////////////////////////
// K MEANS PROJECT EXERCISE ////
///////////////////////////////

// Your task will be to try to cluster clients of a Wholesale Distributor
// based off of the sales of some product categories

// Source of the Data
//http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

// Here is the info on the data:
// 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
// 2)	MILK: annual spending (m.u.) on milk products (Continuous);
// 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
// 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
// 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
// 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
// 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
// 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)

////////////////////////////////////
// COMPLETE THE TASKS BELOW! //////
//////////////////////////////////

// Optional: Use the following code below to set the Error reporting

// Create a Spark Session Instance

// Import Kmeans clustering Algorithm

// Load the Wholesale Customers Data

// Select the following columns for the training set:
// Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
// Cal this new subset feature_data


// Import VectorAssembler and Vectors

// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
// Remember there is no Label column

// Use the assembler object to transform the feature_data
// Call this new data training_data

// Create a Kmeans Model with K=3

// Fit that model to the training_data

// Evaluate clustering by computing Within Set Sum of Squared Errors.

// Shows the result.

package com.sparklearn.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object KMeansProjectExercise {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)
    val spark = SparkSession.builder().getOrCreate()

    val path = getClass.getResource("/Wholesale_customers_data.csv").getPath

    import spark.implicits._

    val dataset = spark.read.option("header", "true").option("inferSchema", value = true).csv(path)

    val feature_data = dataset.select(
      $"Fresh",
      $"Milk",
      $"Grocery",
      $"Frozen",
      $"Detergents_Paper",
      $"Delicassen")

    val vectorAssembler = new VectorAssembler().setInputCols(
      Array(
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen"
      )).setOutputCol("features")

    val trainingData = vectorAssembler.transform(feature_data).select("features")

    val kMeans = new KMeans().setK(3).setSeed(1L)
    val model = kMeans.fit(trainingData)

    val cost = model.computeCost(trainingData)
    println(s"Within Set Sum of Squared Errors = $cost")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }
}




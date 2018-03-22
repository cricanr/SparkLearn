package com.sparklearn.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object PCADocExample {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.SparkConf

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark Learn").getOrCreate()

    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df)

    val pcaDf = pca.transform(df)
    val result = pcaDf.select("pcaFeatures")
    result.show()
  }
}

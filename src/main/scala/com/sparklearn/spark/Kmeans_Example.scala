import com.sparklearn.spark.LogRegExample.getClass
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.clustering.KMeans


object KMeansExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("Your Application Name").setMaster("local")
    val sparkContext = new SparkContext(conf)
    val spark = SparkSession.builder().getOrCreate()

    val path = getClass.getResource("/sample_kmeans_data.txt").getPath

    val dataset = spark.read.option("header", "true").option("inferSchema", value = true).format("libsvm").load(path)

//    val dataset = spark.read.option("header", "true").option("inferSchema", "true").csv("sample_kmeans_data.txt")

    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }
}
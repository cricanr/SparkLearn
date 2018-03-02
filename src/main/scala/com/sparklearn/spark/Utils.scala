package com.sparklearn.spark

import org.apache.spark.sql.DataFrame

object Utils {
  def printData(data: DataFrame): Unit = {
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
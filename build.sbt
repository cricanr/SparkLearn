name := "spark-learn"

organization := "catland"

version := "1.0"

scalaVersion := "2.11.8"

crossScalaVersions := Seq("2.10.4", "2.11.2")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.5" % Test,
  "org.scalacheck" %% "scalacheck" % "1.11.5" % "test",
  "org.apache.spark" %% "spark-core" % "2.3.0",
  "org.apache.spark" %% "spark-sql" % "2.3.0",
  "org.apache.spark" %% "spark-mllib" % "2.3.0",
  "com.github.nscala-time" %% "nscala-time" % "2.18.0"
)

initialCommands := "import example._"

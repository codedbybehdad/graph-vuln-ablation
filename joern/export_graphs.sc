import java.io.File
import java.io.PrintWriter

@main def exec(inputPath: String, outputPath: String) = {

  val srcDir = new File(inputPath)
  val files = srcDir.listFiles.filter(_.getName.endsWith(".c")).toList
  new File(outputPath).mkdirs()

  files.zipWithIndex.foreach { case (f, idx) =>

    println(s"[${idx+1}/${files.size}] ${f.getName}")

    try {

      importCode.c(f.getAbsolutePath)

      val writer = new PrintWriter(
        new File(outputPath, f.getName.replace(".c",".txt"))
      )

      // -------- NODES --------
      writer.println("#NODES")

      cpg.all.foreach { n =>

        val id = n.id()
        val label = n.label

        val code =
          try { n.property("CODE").toString }
          catch { case _: Throwable => "" }

        val clean = code.replaceAll("[\n\r|]", " ")

        writer.println(s"$id|$label|$clean")
      }


      // -------- AST --------
      writer.println("#AST")

      cpg.all.foreach { n =>
        try {
          n._astOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }


      // -------- CFG --------
      writer.println("#CFG")

      cpg.all.foreach { n =>
        try {
          n._cfgOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }


      // -------- DFG --------
      writer.println("#DFG")

      cpg.all.foreach { n =>
        try {
          n._reachingDefOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }


      writer.close()

    } catch {

      case e: Exception =>
        println(s"[ERROR] ${f.getName}: ${e.getMessage}")

    } finally {

      delete

    }

  }
}
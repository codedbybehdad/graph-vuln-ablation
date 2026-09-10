/**
 * Export one graph file per input C function containing exactly three
 * edge families used by the thesis experiments:
 *
 *   AST  = abstract syntax edges
 *   CFG  = control-flow edges
 *   PDG  = control-dependence + data-dependence edges
 *
 * The output is deliberately a simple sectioned text format so that the
 * Python dataset builder can parse it without depending on Joern's DOT
 * representation details.
 */

import java.io.File
import java.io.PrintWriter

@main def exec(inputPath: String, outputPath: String) = {

  val srcDir = new File(inputPath)
  val files = srcDir.listFiles.filter(_.getName.endsWith(".c")).toList.sortBy(_.getName)
  val outDir = new File(outputPath)
  outDir.mkdirs()

  files.zipWithIndex.foreach { case (f, idx) =>

    println(s"[${idx + 1}/${files.size}] ${f.getName}")

    try {
      importCode.c(f.getAbsolutePath)

      // The data-flow layer must exist before reaching-definition edges are read.
      run.ossdataflow

      val outFile = new File(outDir, f.getName.replace(".c", ".txt"))
      val writer = new PrintWriter(outFile)

      writer.println("#NODES")
      cpg.all.foreach { n =>
        val id = n.id()
        val label = n.label
        val code =
          try { n.property("CODE").toString }
          catch { case _: Throwable => "" }
        val clean = code.replaceAll("[\\n\\r|]", " ")
        writer.println(s"$id|$label|$clean")
      }

      writer.println("#AST")
      cpg.all.foreach { n =>
        try {
          n._astOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }

      writer.println("#CFG")
      cpg.all.foreach { n =>
        try {
          n._cfgOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }

      writer.println("#PDG")

      // Control dependence edges are part of the Program Dependence Graph.
      cpg.all.foreach { n =>
        try {
          n._cdgOut.foreach { m =>
            writer.println(s"${n.id()} ${m.id()}")
          }
        } catch { case _: Throwable => }
      }

      // Reaching-definition edges provide the data-dependence component.
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

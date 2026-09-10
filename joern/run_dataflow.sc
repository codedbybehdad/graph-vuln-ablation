@main def exec(cpgFile: String) = {
  importCpg(cpgFile)
  run.ossdataflow
  save
}

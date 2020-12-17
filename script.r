rm(list=ls(all=TRUE))
library("ctbn")


# Read Command Line Arguments
args <- commandArgs(trailingOnly = TRUE)
cat(args, sep = "\n")
folder <- args[1]

# Extract List of samples
all.files <- list.files(folder)
samples = list()
i <- 1
for(fileIter in all.files){
  absPath = file.path(folder, fileIter)
  if(substr(fileIter, nchar(fileIter)-2, nchar(fileIter)) != "var")
  {
    if(substr(fileIter, nchar(fileIter)-4, nchar(fileIter)) != "rctbn")
    {
      s1 = as.data.frame(read.csv(absPath, sep = ";", check.names=FALSE))
      samples[[i]] <- s1
      i <- i + 1
    }
  }
  else
  {
    vars = as.data.frame(read.csv(absPath, sep = ";", check.names=FALSE))
  }
}

# Load Structure from file
ctbn.file <- file.path(folder, "net.rctbn")
xpCtbn3    <- LoadRCtbn(ctbn.file)
dynStr   <- GetDynStruct(xpCtbn3)
stcStr   <- GetBnStruct(xpCtbn3)
print(dynStr)
print(stcStr)


# Learn parameters from samples
print("parameters learning from fully observed data")
LearnCtbnParams(xpCtbn3,samples)

print("delete new ctbn")
xpCtbn2  <- DeleteCtbn(xpCtbn3)
garbage  <- gc()

outputPath <- file.path(folder, "done.rctbn")
SaveRCtbn (xpCtbn3, outputPath)
garbage  <- gc()

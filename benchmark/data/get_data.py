import sys
import subprocess
import numpy as np
from tabulate import tabulate

def get_output(fname):
  cpu = []
  gpu = []
  
  tags_cpu = []
  tags_gpu = []
 
  aux_cpu = []
  aux_gpu = []
  
  f = open(fname, 'r')
  count = 0
  for line in f:
    temp = line.split()
    if( len(temp) == 2):
      if(temp[0][:6] == "CPUGPU"):
        count += 1
        if count % 2 == 1:
	  tags_cpu.append("total")
          aux_cpu.append(float(temp[1]))

          cpu.append(aux_cpu)
          aux_cpu = []
        else:
	  tags_gpu.append("total")
	  aux_gpu.append(float(temp[1]))
 
          gpu.append(aux_gpu)
	  aux_gpu = []
      elif temp[0][:3] == "GPU":
	aux_gpu.append(float(temp[1]))  
        tags_gpu.append(temp[0][4:].split("_")[0])
      elif temp[0][:3] == "CPU":
	aux_cpu.append(float(temp[1]))
        tags_cpu.append(temp[0][4:].split("_")[0])


  val = len(tags_cpu) / len(cpu)
  return tags_gpu[:val], np.array(gpu[1:]), tags_cpu[:val], np.array(cpu[1:]) #ignoring the first row

def show_indiv(header, mp, bs):
  fn = []
  print "For minibatch =", bs
  fn.append(["CPU"] +  mp[bs][1].tolist())
  fn.append(["GPU"] + mp[bs][0].tolist())
  print tabulate(fn, header, tablefmt='pipe')
  
  

def main(filep):
  mp = {}
  bs_array = [1,2,8,32, 64, 128, 256]

  for batchsize in bs_array:
    fname = filep + "_" + str(batchsize) + ".txt"
    t1, g, t2, c = get_output(fname)
    print g
    print c
    mp[batchsize] = [np.mean(g, axis = 0), np.mean(c, axis = 0)]
    print mp[batchsize]
    print

  
  avg_total_inf_cpu = []
  avg_total_inf_gpu = []
  
  for batchsize in bs_array:
    cur_cpu = mp[batchsize][1][-1]   
    cur_gpu = mp[batchsize][0][-1]
    avg_total_inf_cpu.append(cur_cpu)
    avg_total_inf_gpu.append(cur_gpu)

  ## GET AVERGE TIMES FOR TOTAL STATISTICS
  fn = []
  fn.append(avg_total_inf_cpu)
  fn.append(avg_total_inf_gpu)
  fn = np.array(fn)
  header =["mb=" + str(i) for  i in bs_array ]
  print tabulate(fn, header, tablefmt='pipe')

  t1[1] = 'Relu'
  t1[5] = 'Relu'
  ## NOW INDIVIDUAL OPS
  for bs in bs_array:
    show_indiv(t1, mp, bs)
  

if __name__ == "__main__":

  filepref = sys.argv[1]
  main(filepref)



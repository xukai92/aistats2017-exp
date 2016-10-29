# Env setup for Atom
ENV["PATH"] *= ";" * joinpath(JULIA_HOME,"..","Git","usr","bin")

# Helper functions
monitor(kw) = readall(pipeline(`ps aux`, `grep $kw`))
str2vec(str) = filter(s -> s != "", split(str, ' '))

# Constants
HEADERSTR = "USER               PID  %CPU %MEM      VSZ    RSS   TT  STAT STARTED      TIME COMMAND"
HEADER = str2vec(HEADERSTR)
RSSIDX = find(h -> h == "RSS", HEADER)[1]

# Set up monitor time
if length(ARGS) == 2
  KW = ARGS[1]
  MONTIME = int(ARGS[2])
else
  print("Wrong arguments.\n")
  print("Useage: julia mem_monitor.jl [EXEUTABLE] [SEC2MONITOR]")
  exit()
end

# Main
start_t = time()
max_mem = 0
while time() - start_t < MONTIME
  # do sth
  res = split(monitor(KW), '\n')[1:end-1]
  mem = 0
  for line in res
    line_vec = str2vec(line)
    # Exclude process created by this scripts
    if length(find(s -> s == "grep" || s == "julia", line_vec)) == 0
      mem += int(line_vec[RSSIDX])
    end
  end
  if mem > max_mem; max_mem = mem; end
end

print("Maximum memory allocated to $KW is $max_mem within $MONTIME second(s).")

# res = split(monitor("big-hmm"), '\n')[1:end-1][1]
# line_vec = str2vec(res)
# find(s -> s == "grep", line_vec)

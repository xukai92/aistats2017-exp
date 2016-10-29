using Turing, Distributions, Gadfly, DataFrames

include("very-big-hmm.jl")

function run_bm(num_obs, num_part, is_replay)
  global N
  N = num_obs
  total = @elapsed sample(big_hmm, SMC(num_part, is_replay))
  df = readtable("temp.csv", separator = ',', header = false)
  fork = sum([df[1,i] for i = 1:(length(df[1,:])-1)])
  fork / total
end

# Initialise results
copy_ratios = Dict()
replay_ratios = Dict()
POSIX_ratios = Dict()

# Set up experiment parameters
BATCHSIZE = 10
NUMOBS = 75
NUMPARTS = [5, 25, 125]

# Coroutine copying
for num_part = NUMPARTS
  ratios = Vector{Float64}()
  for _ = 1:BATCHSIZE
    push!(ratios, run_bm(NUMOBS, num_part, false))
  end
  copy_ratios[num_part] = ratios
end

# Coroutine replaying
for num_part = NUMPARTS
  ratios = Vector{Float64}()
  for _ = 1:BATCHSIZE
    push!(ratios, run_bm(NUMOBS, num_part, true))
  end
  replay_ratios[num_part] = ratios
end

# POSIX forking
current_path = pwd()
cd("exp1/probc/")

run(pipeline(`make ENGINE=smc`, stdout=DevNull, stderr=DevNull))

for num_part = NUMPARTS
  ratios = Vector{Float64}()

  for _ = 1:BATCHSIZE
    run(pipeline(`./bin/big-hmm -p $num_part`, stdout="total.txt", stderr=DevNull))
    df = readtable("fork.csv", separator = ',', header = false)
    fork = sum([df[1,i] for i = 1:(length(df[1,:])-1)])
    df = readtable("total.txt", header = false)
    total = df[1,1]
    push!(ratios, fork / total)
  end
  POSIX_ratios[num_part] = ratios
end

cd(current_path)

###############
# Making plot #
###############
copy_means = Dict()
replay_means = Dict()
POSIX_means = Dict()

for num_part = NUMPARTS
  copy_means[num_part] = mean(copy_ratios[num_part])
  replay_means[num_part] = mean(replay_ratios[num_part])
  POSIX_means[num_part] = mean(POSIX_ratios[num_part])
end

copy_stds = Dict()
replay_stds = Dict()
POSIX_stds = Dict()

for num_part = NUMPARTS
  copy_stds[num_part] = std(copy_ratios[num_part])
  replay_stds[num_part] = std(replay_ratios[num_part])
  POSIX_stds[num_part] = std(POSIX_ratios[num_part])
end

copy_mins = Dict()
replay_mins = Dict()
POSIX_mins = Dict()

for num_part = NUMPARTS
  copy_mins[num_part] = copy_means[num_part] - (1.96 * copy_stds[num_part] / sqrt(10))
  replay_mins[num_part] = replay_means[num_part] - (1.96 * replay_stds[num_part] / sqrt(10))
  POSIX_mins[num_part] = POSIX_means[num_part] - (1.96 * POSIX_stds[num_part] / sqrt(10))
end

copy_maxs = Dict()
replay_maxs = Dict()
POSIX_maxs = Dict()

for num_part = NUMPARTS
  copy_maxs[num_part] = copy_means[num_part] + (1.96 * copy_stds[num_part] / sqrt(10))
  replay_maxs[num_part] = replay_means[num_part] + (1.96 * replay_stds[num_part] / sqrt(10))
  POSIX_maxs[num_part] = POSIX_means[num_part] + (1.96 * POSIX_stds[num_part] / sqrt(10))
end

errorbar_x = []
append!(errorbar_x, [copy_means[np]  for np = NUMPARTS])
append!(errorbar_x, [replay_means[np]  for np = NUMPARTS])
append!(errorbar_x, [POSIX_means[np]  for np = NUMPARTS])

errorbar_xmin = []
append!(errorbar_xmin, [copy_mins[np]  for np = NUMPARTS])
append!(errorbar_xmin, [replay_mins[np]  for np = NUMPARTS])
append!(errorbar_xmin, [POSIX_mins[np]  for np = NUMPARTS])

errorbar_xmax = []
append!(errorbar_xmax, [copy_maxs[np]  for np = NUMPARTS])
append!(errorbar_xmax, [replay_maxs[np]  for np = NUMPARTS])
append!(errorbar_xmax, [POSIX_maxs[np]  for np = NUMPARTS])

layer_errorbar = layer(
  ymax = [0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 5.5] .+ 0.25,
  ymin = [0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 5.5] .- 0.25,
  x = errorbar_x, y = [0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 5.5],
  xmin = errorbar_xmin, xmax = errorbar_xmax,
  Geom.errorbar, Theme(default_color=colorant"lightsteelblue")
)

label_x = []
append!(label_x, [copy_means[np]  for np = NUMPARTS])
append!(label_x, [replay_means[np]  for np = NUMPARTS])
append!(label_x, [POSIX_means[np]  for np = NUMPARTS])
label_label = []
append!(label_label, ["$(round(copy_means[np] * 100, 2))%"for np = NUMPARTS])
append!(label_label, ["$(round(replay_means[np] * 100, 1))%"  for np = NUMPARTS])
append!(label_label, ["$(round(POSIX_means[np] * 100, 1))%"  for np = NUMPARTS])

layer_label = layer(
  y = [0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 5.5],
  x = label_x,label = label_label, Geom.label(position=:right)
)

label2_label = []
append!(label2_label, [string(np) for np = NUMPARTS])
append!(label2_label, [string(np) for np = NUMPARTS])
append!(label2_label, [string(np) for np = NUMPARTS])
layer_label2 = layer(
  y = [0.5, 1, 1.5, 2.5, 3, 3.5, 4.5, 5, 5.5],
  x = [0, 0, 0, 0, 0, 0, 0, 0, 0],
  label = label2_label,
  Geom.label(position=:left)
)

layer_process_1 = layer(
    ymax = [4.5 + 0.25], ymin = [4.5 - 0.25],
    x = [POSIX_means[NUMPARTS[1]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"skyblue")
)

layer_process_2 = layer(
    ymax = [5 + 0.25], ymin = [5 - 0.25],
    x = [POSIX_means[NUMPARTS[2]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"skyblue3")
)

layer_process_3 = layer(
    ymax = [5.5 + 0.25], ymin = [5.5 - 0.25],
    x = [POSIX_means[NUMPARTS[3]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"skyblue4")
)

# Coroutine replaying
layer_replaying_1 = layer(
    ymax = [2.5 + 0.25], ymin = [2.5 - 0.25],
    x = [replay_means[NUMPARTS[1]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"hotpink")
)

layer_replaying_2 = layer(
    ymax = [3 + 0.25], ymin = [3 - 0.25],
    x = [replay_means[NUMPARTS[2]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"hotpink3")
)

layer_replaying_3 = layer(
    ymax = [3.5 + 0.25], ymin = [3.5 - 0.25],
    x = [replay_means[NUMPARTS[3]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"hotpink4")
)

# Coroutine copying
layer_coroutine_1 = layer(
    ymax = [0.5 + 0.25], ymin = [0.5 - 0.25],
    x = [copy_means[NUMPARTS[1]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"greenyellow")
)

layer_coroutine_2 = layer(
    ymax = [1 + 0.25], ymin = [1 - 0.25],
    x = [copy_means[NUMPARTS[2]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"green3")
)

layer_coroutine_3 = layer(
    ymax = [1.5 + 0.25], ymin = [1.5 - 0.25],
    x = [copy_means[NUMPARTS[3]]],
    Geom.bar(orientation=:horizontal),
    Theme(default_color=colorant"green4")
)

# Plot
p = plot(
  layer_label, layer_label2, layer_errorbar,
  layer_coroutine_1, layer_coroutine_2, layer_coroutine_3,
  layer_replaying_1, layer_replaying_2, layer_replaying_3,
  layer_process_1, layer_process_2, layer_process_3,
  Stat.yticks(ticks=[0, 1, 2, 3, 4]),
  Stat.xticks(ticks=[0.00, 0.25, 0.50, 0.75, 1.00]),
  Guide.title("Percentage of Times Spent on Program State Copying"),
  Guide.manual_color_key("",["POSIX Forking   ","Coroutine Replaying   ", "Coroutine Copying"], ["skyblue3", "hotpink3", "green3"]),
  Theme(key_position=:top, key_label_font_size=9.5pt),
  Guide.yticks(label=false),
  Guide.ylabel("Number of Particles"),
  Guide.xlabel("Program State Copying / Total (%)"),
  Coord.Cartesian(xmin=-0.05,xmax=1.00),
  Scale.x_continuous(labels=x->"$(round(Int, x*100))%")
)

# Save plot
draw(PDF("exp1.pdf", 5inch, 3.75inch), p)

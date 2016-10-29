# Scripts to find memory allocation of SMC with coroutine copying or replaying

using Turing
using Distributions
include("../very-big-hmm.jl")
N = 75  # number of obserations

copy_mems = []
for _ = 1:10
    mem = @allocated sample(big_hmm, SMC(100, false))
    push!(copy_mems, mem)
end

replay_mems = []
for _ = 1:10
    mem = @allocated sample(big_hmm, SMC(100, true))
    push!(replay_mems, mem)
end

println("Copy: $(round(mean(copy_mems)/ 1024 / 1024, 3)) MB")
println("Replay: $(round(mean(replay_mems) / 1024 / 1024, 3)) MB")
POSIX_mems = [
  70644, 65256, 58024, 75200, 91472,
  78816, 75208, 74296, 75200, 84240
] # manual running result
println("POSIX: $(round(mean(POSIX_mems) / 1024, 3)) MB")

# Results obtained on 29 Oct
# Single core on my local macOS machine
# Copy: 20.068 MB
# Replay: 37.348 MB
# POSIX: 73.082 MB

push!(LOAD_PATH, (Base.@__DIR__) * "/src")

using MPSLearning


L = 10
d = 2
N = 50


training_x = [renormalize!(createrandommps(L, d, 1)) for i in 1:N]
training_y = [renormalize!(createrandommps(L, d, 1)) for i in 1:N]

dmrg = OptimizeMPO(training_x, training_y, alpha=0.01, D=4)

for i in 1:5
    dosweep!(dmrg, verbose=2)
end

testing_x = [renormalize!(createrandommps(L, d, 1)) for i in 1:5]

predicted_y = [predict(dmrg, item) for item in testing_x]

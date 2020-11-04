module MPSLearning

export generateprodmps, createrandommps, renormalize!, physical_dimensions, createrandommpo
export OptimizeMPO, dosweep!, compute!, predict

import Base.*
using LinearAlgebra: qr!, tr, norm, dot, diagm

include("tensorop.jl")
include("mps.jl")
include("mpo.jl")
include("seq2seq.jl")



end

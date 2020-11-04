
struct NSymMPO{T}
	data::Vector{Array{T, 4}}
end

data(x::NSymMPO) = x.data
Base.eltype(x::NSymMPO) = eltype(data(x))
Base.getindex(x::NSymMPO, i::Int) = getindex(data(x), i)
Base.setindex!(x::NSymMPO, v,  i::Int) = setindex!(data(x), v, i)
Base.length(x::NSymMPO) = length(data(x))
Base.isempty(x::NSymMPO) = isempty(data(x))


NSymMPO{T}() where T = NSymMPO(Vector{Array{T, 4}}())
NSymMPO{T}(L::Int) where T = NSymMPO(Vector{Array{T, 4}}(undef, L))
Base.conj(a::NSymMPO) = NSymMPO(conj(data(a)))

function _compute_mpo_D(L::Int, dx::Int, dy::Int, D::Int)
	Ds = [1 for i=1:(L+1)]
	Lhalf = div(L, 2)
	for i = 1:Lhalf
		Ds[i+1] = min(D, Ds[i]*dx*dy)
	end
	for i = L:-1:(Lhalf+1)
		Ds[i] = min(D, Ds[i+1]*dx*dy)
	end
	return Ds
end

"""
	createrandommpo(::Type{T}, L::Int, dx::Int, dy::Int, D::Int) where {T <: Number}
Return a non-symmetric MPO
"""
function createrandommpo(::Type{T}, L::Int, dx::Int, dy::Int, D::Int) where {T <: Number}
	(L <= 1) && error("createrandommpo require L larger than 1.")
	mpo = NSymMPO{T}(L)
	Ds = _compute_mpo_D(L, dx, dy, D)
	for i in 1:L
		mpo[i] = randn(T, dy, Ds[i], Ds[i+1], dx)/D
	end
	return mpo
end

"""
	createrandommpo(L::Int, dx::Int, dy::Int, D::Int)
Return a non-symmetric MPO
"""
createrandommpo(L::Int, dx::Int, dy::Int, D::Int) = createrandommpo(Float64, L, dx, dy, D)

"""
	createrandommpo(::Type{T}, dx::Vector{Int}, dy::Vector{Int}, D::Int) where {T <: Number}
Return a non-symmetric MPO
"""
function createrandommpo(::Type{T}, dx::Vector{Int}, dy::Vector{Int}, D::Int) where {T <: Number}
	(length(dx) != length(dy)) && error("dx, dy size mismatch.")
	(length(dx) <= 1) && error("createrandommpo require L larger than 1.")
	L = length(dx)
	mpo = NSymMPO{T}(L)
	mpo[1] = randn(T, dy[1], 1, D, dx[1])
	mpo[L] = randn(T, dy[L], D, 1, dx[L])
	for i = 2:(L-1)
		mpo[i] = randn(T, dy[i], D, D, dx[i])
	end
	return mpo
end

"""
	createrandommpo(dx::Vector{Int}, dy::Vector{Int}, D::Int)
Return a non-symmetric MPO
"""
createrandommpo(dx::Vector{Int}, dy::Vector{Int}, D::Int) = createrandommpo(Float64, dx, dy, D)


"""
	*(mpo::AbstractMPO, mps::AbstractMPS)
Multiplication mpo with mps.
"""
function *(mpo::NSymMPO{T}, mps::NSymMPS{T}) where T
	(length(mpo) != length(mps)) && error("dot mps requires mpo and mps of same size.")
	isempty(mpo) && error("mpo is empty.")
	L = length(mps)
	# res = promote_type(typeof(mpo), typeof(mps))(L)
	res = Vector{Any}(undef, L)
	for i = 1:L
	    res[i] = contract(mpo[i], mps[i], ((4,), (2,)))
	end
	res[1], temp = fusion(res[1], nothing, ((2,4), (1,1)))
	res[1] = permute(res[1], (4,1,2,3))
	for i=1:(L-1)
	    res[i], res[i+1] = fusion(res[i], res[i+1], ((3,4), (2,4)))
	end
	res[L], temp = fusion(res[L], nothing, ((3,4),(1,1)))
	return NSymMPS([Array(item) for item in res])
end

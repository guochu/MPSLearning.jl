


struct NSymMPS{T}
	data::Vector{Array{T, 3}}
end

data(x::NSymMPS) = x.data
Base.eltype(x::NSymMPS) = eltype(data(x))
Base.getindex(x::NSymMPS, i::Int) = getindex(data(x), i)
Base.setindex!(x::NSymMPS, v,  i::Int) = setindex!(data(x), v, i)
Base.length(x::NSymMPS) = length(data(x))
Base.isempty(x::NSymMPS) = isempty(data(x))

Base.iterate(x::NSymMPS) = iterate(data(x))
Base.iterate(x::NSymMPS, state) = iterate(data(x), state)
Base.IndexStyle(::Type{<:NSymMPS}) = IndexLinear()
Base.firstindex(x::NSymMPS) = firstindex(data(x))
Base.lastindex(x::NSymMPS) = lastindex(data(x))

NSymMPS{T}() where T = NSymMPS(Vector{Array{T, 3}}())
NSymMPS{T}(L::Int) where T = NSymMPS(Vector{Array{T, 3}}(undef, L))
Base.conj(a::NSymMPS) = NSymMPS(conj(data(a)))

"""
	physical_dimensions(mps::AbstractNSymMPS)
Return physical dimensions of mps
"""
physical_dimensions(mps::NSymMPS) = [shape(s, 2) for s in mps]

function _compute_mps_D(d::Vector{Int}, D::Int)
	(length(d)==1) && return [1, 1]
	L = length(d)
	Ds = [1 for i =1:(L+1)]
	Lhalf = div(L, 2)
	for i in 1:Lhalf
		Ds[i+1] = min(D, Ds[i]*d[i])
	end
	for i in L:-1:(Lhalf+1)
		Ds[i] = min(D, Ds[i+1]*d[i])
	end
	return Ds
end

"""
	createrandommps(::Type{T}, d::Vector{Int}, D::Int) where {T <: Number}
Return a random non-symmetric mps.
"""
function createrandommps(::Type{T}, d::Vector{Int}, D::Int) where {T <: Number}
	isempty(d) && error("d must not be empty for createrandommps.")
	L = length(d)
	mps = NSymMPS{T}(L)
	Ds = _compute_mps_D(d, D)
	for i = 1:L
		mps[i] = randn(T, Ds[i], d[i], Ds[i+1])
		mps[i] /= sqrt(length(mps[i]))
	end
	return mps
end

"""
	createrandommps(d::Vector{Int}, D::Int)
Return a random non-symmetric mps.
"""
createrandommps(d::Vector{Int}, D::Int) = createrandommps(Float64, d, D)

"""
	createrandommps(::Type{T}, L::Int, d::Int, D::Int) where {T <: Number}
Return a random non-symmetric mps.
"""
createrandommps(::Type{T}, L::Int, d::Int, D::Int) where {T <: Number} = createrandommps(T, [d for i in 1:L], D)

"""
	createrandommps(L::Int, d::Int, D::Int)
Return a random non-symmetric mps.
"""
createrandommps(L::Int, d::Int, D::Int) = createrandommps(Float64, L, d, D)


"""
	generateprodmps(::Type{T}, ds::Vector{Int}, mpsstr::Vector{Int}) where T
Return a product non-symmetric mps.
"""
function generateprodmps(::Type{T}, ds::Vector{Int}, mpsstr::Vector{Int}) where T
	(length(ds) != length(mpsstr)) && error("size mismatch for generateprodmps.")
	L = length(ds)
	mps = NSymMPS{T}(L)
	for i = L:-1:1
		d = ds[i]
		mps[i] = zeros(T,1,d,1)
		mps[i][1, mpsstr[i]+1, 1] = 1.
	end
	return mps
end

"""
	generateprodmps(ds::Vector{Int}, mpsstr::Vector{Int})
Return a product non-symmetric mps.
"""
generateprodmps(ds::Vector{Int}, mpsstr::Vector{Int}) = generateprodmps(Float64, ds, mpsstr)


"""
	generateprodmps(::Type{T}, ds::Vector{Int}, mpsstr::Vector{Vector{T1}}) where {T<:Number, T1<:Number}
Return a product non-symmetric mps.
"""
function generateprodmps(::Type{T}, ds::Vector{Int}, mpsstr::Vector{Vector{T1}}) where {T<:Number, T1<:Number}
	(length(ds) != length(mpsstr)) && error("size mismatch for generateprodmps.")
	L = length(ds)
	mps = NSymMPS{T}(L)
	for i = L:-1:1
		d = ds[i]
		(length(mpsstr[i]) != d) && error("size mismatch for sitemps.")
		# mps[i] = zeros(T,1,d,1)
		# mps[i][1, :, 1] = mpsstr[i]
		mps[i] = reshape(mpsstr[i], 1, d, 1)
	end
	return mps
end

# """
# 	generateprodmps(ds::Vector{Int}, mpsstr::Vector{Vector{T}}) where {T<:Number}
# Return a product non-symmetric mps.
# """
# generateprodmps(ds::Vector{Int}, mpsstr::Vector{Vector{T}}) where {T<:Number} = generateprodmps(T, ds, mpsstr)

"""
	generateprodmps(mpsstr::Vector{Vector{<:Number}})
Return a product non-symmetric mps.
"""
generateprodmps(::Type{T}, mpsstr::Vector{Vector{T1}}) where {T<:Number, T1<:Number} = generateprodmps(T, length.(mpsstr), mpsstr)
generateprodmps(mpsstr::Vector{Vector{T}}) where {T<:Number} = generateprodmps(T, mpsstr)

updateCrighth1h2(hold::Nothing, obj::Nothing, mpsAj::AbstractArray, mpsBj::AbstractArray) = contract(
	mpsBj, mpsAj, ((2,3),(2,3)))

function updateCrighth1h2(hold::Nothing, obj::AbstractArray, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(obj, mpsAj, ((2,),(2,)))
	return contract(mpsBj, Hnew, ((2,3),(1,3)))
end

function updateCrighth1h2(hold::AbstractArray, obj::Nothing, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((3,),(1,)))
	return contract(Hnew, mpsAj, ((2,3),(2,3)))
end

function updateCrighth1h2(hold::AbstractArray, obj::AbstractArray, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((3,),(1,)))
	Hnew = contract(Hnew, obj, ((2,),(1,)))
	return contract(Hnew, mpsAj, ((3,2),(2,3)))
end

"""
	renormalize!(x::AbstractMPS)
renormalize MPS to be norm 1.
"""
function renormalize!(x::NSymMPS)
	hold = nothing
	for i = length(x):-1:1
	    hold = updateCrighth1h2(hold, nothing, conj(x[i]), x[i])
	    if (i%3 == 0)
	    	s = norm(hold)
	    	x[i] /= s
	    	hold /= (s*s)
	    end
	end
	sc = tr(hold)
	s = sqrt(real(sc))
	x[1] /= s
	return x
end

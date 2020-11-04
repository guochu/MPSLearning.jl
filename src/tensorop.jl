
"""
	move_selected_index_forward(a, I)
	move the indexes specified by I to the front of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_forward(a::Vector{T}, I) where {T}
    na = length(a)
    nI = length(I)
    b = Vector{T}(undef, na)
    k1 = 0
    k2 = nI
    for i=1:na
        s = 0
        while s != nI
        	if i == I[s+1]
        		b[s+1] = a[k1+1]
        	    k1 += 1
        	    break
        	end
        	s += 1
        end
        if s == nI
        	b[k2+1]=a[k1+1]
        	k1 += 1
            k2 += 1
        end
    end
    return b
end

function move_selected_index_forward(a::NTuple{N, T}, I) where {N, T}
    return NTuple{N, T}(move_selected_index_forward([a...], I))
end

"""
	move_selected_index_backward(a, I)
	move the indexes specified by I to the back of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_backward(a::Vector{T}, I) where {T}
	na = length(a)
	nI = length(I)
	nr = na - nI
	b = Vector{T}(undef, na)
	k1 = 0
	k2 = 0
	for i = 1:na
	    s = 0
	    while s != nI
	    	if i == I[s+1]
	    		b[nr+s+1] = a[k1+1]
	    		k1 += 1
	    		break
	    	end
	    	s += 1
	    end
	    if s == nI
	        b[k2+1] = a[k1+1]
	        k2 += 1
	        k1 += 1
	    end
	end
	return b
end

function move_selected_index_backward(a::NTuple{N, T}, I) where {N, T}
	return NTuple{N, T}(move_selected_index_backward([a...], I))
end

permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)

diag(m::AbstractArray{T, 1}) where T = diagm(0=>m)
eye(::Type{T}, d::Int) where T = diag(ones(T, d))
eye(d::Int) = eye(Float64, d)

shape(m::AbstractArray) = size(m)
shape(m::AbstractArray, i::Int) = size(m, i)


# do we really need tie function?
function _group_extent(extent::NTuple{N, Int}, idx::NTuple{N1, Int}) where {N, N1}
    ext = Vector{Int}(undef, N1)
    l = 0
    for i=1:N1
        ext[i] = prod(extent[(l+1):(l+idx[i])])
        l += idx[i]
    end
    return NTuple{N1, Int}(ext)
end


function tie(a::AbstractArray{T, N}, axs::NTuple{N1, Int}) where {T, N, N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank.")
    return reshape(a, _group_extent(shape(a), axs))
end

"""
    contract(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
Dense tensor contract
"""
function contract(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
    ia, ib = axs
    seqindex_a = move_selected_index_backward(collect(1:Na), ia)
    seqindex_b = move_selected_index_forward(collect(1:Nb), ib)
    ap = permute(a, seqindex_a)
    bp = permute(b, seqindex_b)
    return reshape(tie(ap, (Na-N, N)) * tie(bp, (N, Nb-N)), shape(ap)[1:(Na-N)]..., shape(bp)[(N+1):Nb]...)
end


"""
    qr(a::AbstractArray{T, N}, axs::Tuple{NTuple{N1, Int}, NTuple{N2, Int}}) where {T, N, N1, N2}
QR decomposition of QTensor a, by joining axs to be the second dimension
"""
function qr(a::AbstractArray{T, N}, axs::Tuple{NTuple{N1, Int}, NTuple{N2, Int}}) where {T, N, N1, N2}
    # ranka = rank(a)
    (N == N1+N2) || error("dimension error.")
    ix, iy = axs
    newindex = (ix..., iy...)
    a1 = permute(a, newindex)
    shape_a = shape(a1)
    dimu = shape_a[1:N1]
    s1 = prod(dimu)
    dimv = shape_a[(N1+1):end]
    s2 = prod(dimv)
    F = qr!(Base.Matrix(reshape(a1, s1, s2)))
    u = Base.Matrix(F.Q)
    v = Base.Matrix(F.R)
    s = shape(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    qr(a::AbstractArray{T, N}, axes::NTuple{N1, Int}=(1,)) where {T, N, N1}
QR decomposition of QTensor a, by joining axs to be the second dimension
"""
function qr(a::AbstractArray{T, N}, axs::NTuple{N1, Int}=(1,)) where {T, N, N1}
    # ranka = rank(a)
    newindex = move_selected_index_backward([i for i =1:N], axs)
    return qr(a, (Tuple(newindex[1:(N-N1)]), axs))
end

"""
    fusion(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
fusion of two QTensors with axs and fuse_func
"""
function fusion(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
    IA, IB = axs
    (isempty(a) || isempty(b)) && error("The input a can not be empty.")
    (N > Na) && error("length of axes is larger than tensor rank.")
    indexa = move_selected_index_backward([i for i=1:Na], IA)
    a1 = permute(a, indexa)
    shape_a = shape(a1)
    sizem = prod(shape_a[(Na-N+1):end])
    a1 = reshape(a1, (shape_a[1:(Na-N)]..., sizem))
    b1 = b
    if !isempty(b)
        (N > Nb) && error("length of axes is larger than tensor rank.")
        indexb = move_selected_index_forward([i for i=1:Nb], IB)
        b1 = permute(b1, indexb)
        shape_b = shape(b1)
        sizem = prod(shape_b[1:N])
        b1 = reshape(b1, (sizem, shape_b[(N+1):end]...))
    end
    return a1, b1
end

function fusion(a::AbstractArray{Ta, Na}, b::Nothing, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, N}
    IA, IB = axs
    isempty(a) && error("The input a can not be empty.")
    (N > Na) && error("length of axes is larger than tensor rank.")
    indexa = move_selected_index_backward([i for i=1:Na], IA)
    a1 = permute(a, indexa)
    shape_a = shape(a1)
    sizem = prod(shape_a[(Na-N+1):end])
    (sizem == 1) || error("non-contracted indexes must be trivial.")
    a1 = reshape(a1, (shape_a[1:(Na-N)]..., sizem))
    return a1, nothing
end

function fusion(a::Nothing, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Tb, Nb, N}
    IA, IB = axs
    isempty(b) && error("The input b can not be empty.")
    (N > Nb) && error("length of axes is larger than tensor rank.")
    indexb = move_selected_index_forward([i for i=1:Nb], IB)
    b1 = permute(b, indexb)
    shape_b = shape(b1)
    sizem = prod(shape_b[1:N])
    (sizem == 1) || error("non-contracted indexes must be trivial.")
    b1 = reshape(b1, (sizem, shape_b[(N+1):end]...))
    return nothing, b1
end


function updateCright(hold::AbstractArray, mpoj::Nothing, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((3,),(1,)))
	return contract(Hnew, mpsAj, ((2,4),(2,3)))
end

function updateCright(hold::AbstractArray, mpoj::AbstractArray, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((3,),(1,)))
	Hnew = contract(Hnew, mpoj, ((2,3),(1,3)))
	return contract(Hnew, mpsAj, ((4,2),(2,3)))
end

# update heff from left using mpo
function updateCleft(hold::AbstractArray, mpoj::Nothing, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(hold, mpsAj, ((3,),(1,)))
	return contract(mpsBj, Hnew, ((1,2),(1,3)))
end

function updateCleft(hold::AbstractArray, mpoj::AbstractArray, mpsAj::AbstractArray, mpsBj::AbstractArray)
	Hnew = contract(hold, mpsAj, ((3,),(1,)))
	Hnew = contract(mpoj, Hnew, ((2,4),(2,3)))
	return contract(mpsBj, Hnew, ((1,2),(3,1)))
end

function updateCCleft(hold::AbstractArray, mpsBj::AbstractArray, mpoj1, mpoj2, mpsAj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((1,), (1,)))
	if mpoj1 == nothing
		if mpoj2 == nothing
		    return contract(Hnew, mpsAj, ((5,1), (1,2)))
		else
			Hnew = contract(Hnew, mpoj2, ((1,4), (1,2)))
			return contract(Hnew, mpsAj, ((3,5), (1,2)))
		end
	else
		Hnew = contract(Hnew, mpoj1, ((1,3), (4,2)))
		if mpoj2 == nothing
			Hnew = contract(Hnew, mpsAj, ((3,4), (1,2)))
			return permute(Hnew, (1,3,2,4))
		else
			Hnew = contract(Hnew, mpoj2, ((4,2), (1,2)))
			return contract(Hnew, mpsAj, ((2,5), (1,2)))
		end
	end
end

function updateCCright(hold::AbstractArray, mpsBj::AbstractArray, mpoj1, mpoj2, mpsAj::AbstractArray)
	Hnew = contract(mpsBj, hold, ((3,), (1,)))
	if mpoj1 == nothing
		if mpoj2 == nothing
			return contract(Hnew, mpsAj, ((2,5), (2,3)))
		else
			Hnew = contract(Hnew, mpoj2, ((2,4), (1,3)))
			return contract(Hnew, mpsAj, ((5,3), (2,3)))
		end
	else
		Hnew = contract(Hnew, mpoj1, ((2,3), (4,3)))
		if mpoj2 == nothing
			Hnew = contract(Hnew, mpsAj, ((4,3),(2,3)))
			return permute(Hnew, (1,3,2,4))
		else
			Hnew = contract(Hnew, mpoj2, ((4,2),(1,3)))
			return contract(Hnew, mpsAj, ((5,2),(2,3)))
		end
	end
end

updateCCright(hold::AbstractArray, mpoj::AbstractArray, mpsj::AbstractArray) = updateCCright(
	hold, conj(mpsj), conj(mpoj), mpoj, mpsj)
updateCCleft(hold::AbstractArray, mpoj::AbstractArray, mpsj::AbstractArray) = updateCCleft(
	hold, conj(mpsj), conj(mpoj), mpoj, mpsj)

updateMPOleft(hold::Nothing, mpojA::AbstractArray, mpojB::AbstractArray) = contract(mpojB, mpojA, ((1,2,4), (1,2,4)))
updateMPOleft(hold::Nothing, mpoj::AbstractArray) = contract(conj(mpoj), mpoj, ((1,2,4), (1,2,4)))

function updateMPOleft(hold::AbstractArray, mpojA::AbstractArray, mpojB::AbstractArray)
	hnew = contract(hold, mpojA, ((2,), (2,)))
	return contract(mpsjB, hnew, ((1,2,4), (2,1,4)))
end

function updateMPOleft(hold::AbstractArray, mpoj::AbstractArray)
	hnew = contract(hold, mpoj, ((2,), (2,)))
	return contract(conj(mpoj), hnew, ((1,2,4), (2,1,4)))
end

updateMPOright(hold::Nothing, mpojA::AbstractArray, mpojB::AbstractArray) = contract(mpojB, mpojA, ((1,3,4), (1,3,4)))
updateMPOright(hold::Nothing, mpoj::AbstractArray) = contract(conj(mpoj), mpoj, ((1,3,4), (1,3,4)))

function updateMPOright(hold::AbstractArray, mpojA::AbstractArray, mpojB::AbstractArray)
	hnew = contract(mpojB, hold, ((3,), (1,)))
	return contract(hnew, mpojA, ((1,3,4),(1,4,3)))
end
function updateMPOright(hold::AbstractArray, mpoj::AbstractArray)
	hnew = contract(conj(mpoj), hold, ((3,), (1,)))
	return contract(hnew, mpoj, ((1,3,4),(1,4,3)))
end

"""
	output
	*******1*******
	***2---M----4
	*******3*******
"""
function HSingleSite(mpsxj, mpsyj, hleft, hright)
	hnew = contract(hleft, mpsxj, ((3,),(1,)))
	hnew = contract(hnew, hright, ((4,),(3,)))
	hnew = contract(mpsyj, hnew, ((1,3),(1,4)))
	return hnew
end

"""
	output
	*******1*******
	***2---M----5
	***3---M----6
	*******4*******
"""
function HHSingleSite(mpsj, hleft, hright)
	hnew = contract(conj(mpsj), hleft, ((1,), (1,)))
	hnew = contract(hnew, mpsj, ((5,), (1,)))
	hnew = contract(hnew, hright, ((2,6),(1,4)))
	return hnew
end

function _compute_distance(heff, xeff, mpoj)
	distance = dot(mpoj, heff*mpoj)
	tmp = dot(mpoj, xeff)
	return real(distance) - 2*real(tmp)
end

_mse_distance(dis::Real, ynorm::Real, N::Int) = (dis + ynorm)/N

function initHstorageRight(lbompo, mpsx::NSymMPS{T}, mpsy::NSymMPS{T}) where T
	(length(lbompo)==length(mpsx) && length(lbompo)==length(mpsy)) || error("lbompo size mismatch mpsx, mpsy.")
	L = length(lbompo)
	(L<=1) && error("lbompo size must be larger than 1.")
	hstorage = Vector{Array{T, 3}}(undef, L+1)
	hstorage[1] = ones(1,1,1)
	hstorage[L+1] = ones(1,1,1)
	for i in L:-1:2
	    hstorage[i] = updateCright(hstorage[i + 1], lbompo[i], mpsx[i], conj(mpsy[i]))
	end
	return hstorage
end

"""
	update Hstorage from left till site
"""
function computeHstorageLeft(mpo, mpsx::NSymMPS, mpsy::NSymMPS, site::Int)
	(length(mpo)==length(mpsx) && length(mpo)==length(mpsy)) || error("mpo and mps size mismatch.")
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	(site > L || site < 1) && error("site out of range.")
	h = ones(1,1,1)
	for i = 1:(site-1)
		h = updateCleft(h, mpo[i], mpsx[i], conj(mpsy[i]))
	end
	return h
end

"""
	update Hstorage from right till site-1
"""
function computeHstorageRight(mpo, mpsx::NSymMPS, mpsy::NSymMPS, site::Int)
	(length(mpo)==length(mpsx) && length(mpo)==length(mpsy)) || error("mpo and mps size mismatch.")
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	(site > L || site < 1) && error("site out of range.")
	h = ones(1,1,1)
	for i = L:-1:(site+1)
		h = updateCright(h, mpo[i], mpsx[i], conj(mpsy[i]))
	end
	return h
end

function initHHstorageRight(mpo, mps::NSymMPS{T}) where T
	(length(mpo)==length(mps)) || error("mpo and mps size mismatch.")
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	hstorage = Vector{Array{T, 4}}(undef, L+1)
	hstorage[1] = ones(1,1,1,1)
	hstorage[L+1] = ones(1,1,1,1)
	for i = L:-1:2
		hstorage[i] = updateCCright(hstorage[i+1], mpo[i], mps[i])
	end
	return hstorage
end

function computeHHstorageLeft(mpo, mps::NSymMPS, site::Int)
	(length(mpo)==length(mps)) || error("mpo and mps size mismatch.")
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	(site > L || site < 1) && error("site out of range.")
	h = ones(1,1,1,1)
	for i = 1:(site-1)
		h = updateCCleft(h, mpo[i], mps[i])
	end
	return h
end

function computeHHstorageRight(mpo, mps::NSymMPS, site::Int)
	(length(mpo)==length(mps)) || error("mpo and mps size mismatch.")
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	(site > L || site < 1) && error("site out of range.")
	h = ones(1,1,1,1)
	for i = L:-1:(site+1)
		h = updateCCright(h, mpo[i], mps[i])
	end
	return h
end

function initMPOstorageRight(mpo::NSymMPO{T}) where T
	L = length(mpo)
	(L <= 1) && error("size of mpo must be larger than 1.")
	hstorage = Vector{Array{T, 2}}(undef, L+1)
	hstorage[1] = ones(1,1)
	hstorage[L+1] = ones(1,1)
	for i = L:-1:2
		hstorage[i] = updateMPOright(hstorage[i+1], mpo[i])
	end
	return hstorage
end

struct OptimizeMPO{T}
	mpo::NSymMPO{T}
	mpsxs::Vector{NSymMPS{T}}
	mpsys::Vector{NSymMPS{T}}
	hms::Vector{Array{T, 2}}
	hs::Vector{Vector{Array{T, 3}}}
	hhs::Vector{Vector{Array{T, 4}}}
	alpha::Float64
	D::Int
	N::Int
	L::Int
	dx::Vector{Int}
	dy::Vector{Int}
	kvals::Vector{Float64}
	ynorm::Float64

	function OptimizeMPO(mpsxs::Vector{NSymMPS{T}}, mpsys::Vector{NSymMPS{T}}; alpha::Real=0.01, D::Int=5) where {T <: Number}
		isempty(mpsxs) && error("number of dataset must be larger than 0.")
		N = length(mpsxs)
		(length(mpsys)==N) || error("input x, y number mismatch.")
		L = length(mpsxs[1])
		dx = physical_dimensions(mpsxs[1])
		dy = physical_dimensions(mpsys[1])
		for i = 1:N
			(length(mpsxs[i])==L && length(mpsys[i])==L) || error("input x, y size mismatch.")
			(physical_dimensions(mpsxs[i])==dx) || error("input x physical dimension mismatch.")
			(physical_dimensions(mpsys[i])==dy) || error("input y physical dimension mismatch.")
		end
		mpo = createrandommpo(T, dx, dy, D)
		hms = initMPOstorageRight(mpo)
		hs = [initHstorageRight(conj(mpo), conj(mpsxs[j]), conj(mpsys[j])) for j = 1:N]
		hhs = [initHHstorageRight(mpo, mpsxs[j]) for j = 1:N]
		kvals = Vector{Float64}()
		new{T}(mpo, mpsxs, mpsys, hms, hs, hhs, alpha, D, N, L, dx, dy, kvals, _compute_y_norms(mpsys))
	end
end

get_mpo(s::OptimizeMPO) = s.mpo
get_mpsxs(s::OptimizeMPO) = s.mpsxs
get_mpsys(s::OptimizeMPO) = s.mpsys
get_mpostorage(s::OptimizeMPO) = s.hms
get_alpha(s::OptimizeMPO) = s.alpha
get_D(s::OptimizeMPO) = s.D
get_N(s::OptimizeMPO) = s.N
get_L(s::OptimizeMPO) = s.L
get_dx(s::OptimizeMPO) = s.dx
get_dy(s::OptimizeMPO) = s.dy
get_kvals(s::OptimizeMPO) = s.kvals
get_ynorm(s::OptimizeMPO) = s.ynorm

get_hstorage(s::OptimizeMPO) = s.hs
get_hhstorage(s::OptimizeMPO) = s.hhs


function _compute_eff_impl(s::OptimizeMPO{T}, site::Int, direction::Char) where T
	mpo = get_mpo(s)
	mpsxs = get_mpsxs(s)
	mpsys = get_mpsys(s)
	hhs = get_hhstorage(s)
	hs = get_hstorage(s)
	l = shape(mpo[site], 2)
	r = shape(mpo[site], 3)
	dy = get_dy(s)[site]
	dx = get_dx(s)[site]
	xeff = zeros(T, dy, l, dx, r)
	heff = zeros(T, dx, l, l, dx, r, r)
	for n = 1:get_N(s)
		heff += HHSingleSite(mpsxs[n][site], hhs[n][site], hhs[n][site+1])
		xeff += HSingleSite(conj(mpsxs[n][site]), mpsys[n][site], hs[n][site], hs[n][site+1])
	end
	return heff, xeff
end

function _compute_eff!(s::OptimizeMPO, site::Int, direction::Char)
	hms = get_mpostorage(s)
	mpo = get_mpo(s)
	dx = get_dx(s)[site]
	heff, xeff = _compute_eff_impl(s, site, direction)
	heff = permute(heff, (2,1,5,3,4,6))
	mpoeff = contract(hms[site], hms[site+1], ((), ()))
	mpoeff = contract(mpoeff, eye(dx), ((), ()))
	mpoeff = permute(mpoeff, (1,5,3,2,6,4))
	heff += get_alpha(s)*mpoeff
	xeff = permute(xeff, (2,3,4,1))
	d = shape(heff,1)*shape(heff,2)*shape(heff,3)

	xeff2 = reshape(xeff, (d, shape(xeff,4)))
	heff2 = reshape(heff,(d, d))
	mpoj = heff2\xeff2

	distance = _compute_distance(heff2, xeff2, mpoj)
	mpoj = reshape(mpoj, shape(xeff))
	mpoj = permute(mpoj, (4,1,3,2))

	if direction == 'L'
		mpoj, u = qr(mpoj, (3,))
		mpo[site] = permute(mpoj, (1,2,4,3))
		u = contract(u, mpo[site+1], ((2,), (2,)))
		mpo[site+1] = permute(u, (2,1,3,4))
	else
		mpoj, u = qr(mpoj, (2,))
		mpo[site] = permute(mpoj, (1,4,2,3))
		u = contract(mpo[site-1], u, ((3,), (2,)))
		mpo[site-1] = permute(u, (1,2,4,3))
	end
	return _mse_distance(distance, get_ynorm(s), get_N(s))
end


function _updateStorageLeft!(s::OptimizeMPO, site::Int)
	hs = get_hstorage(s)
	hhs = get_hhstorage(s)
	mpo = get_mpo(s)
	mpsxs = get_mpsxs(s)
	mpsys = get_mpsys(s)
	hms = get_mpostorage(s)
	for n = 1:get_N(s)
		hs[n][site+1] = updateCleft(hs[n][site], conj(mpo[site]), conj(mpsxs[n][site]), mpsys[n][site])
		hhs[n][site+1] = updateCCleft(hhs[n][site], mpo[site], mpsxs[n][site])
	end
	hms[site+1] = updateMPOleft(hms[site], mpo[site])
end


function _updateStorageRight!(s::OptimizeMPO, site::Int)
	hs = get_hstorage(s)
	hhs = get_hhstorage(s)
	mpo = get_mpo(s)
	mpsxs = get_mpsxs(s)
	mpsys = get_mpsys(s)
	hms = get_mpostorage(s)
	for n = 1:get_N(s)
		hs[n][site] = updateCright(hs[n][site+1], conj(mpo[site]), conj(mpsxs[n][site]), mpsys[n][site])
		hhs[n][site] = updateCCright(hhs[n][site+1], mpo[site], mpsxs[n][site])
	end
	hms[site] = updateMPOright(hms[site+1], mpo[site])
end


function oneSiteSweepLeft_impl!(s::OptimizeMPO; verbose::Int=1)
	kvals = Vector{Float64}()
	L = get_L(s)
	for j = 1:(L-1)
		(verbose > 2) && println("One site sweep from left to right on site $j.")
		distance = _compute_eff!(s, j, 'L')
		push!(kvals, distance)
		(verbose >= 2) && println("loss after updating: $distance.")
		_updateStorageLeft!(s, j)
	end
	append!(get_kvals(s), kvals)
	return kvals
end

oneSiteSweepLeft!(s::OptimizeMPO; verbose::Int=1) = oneSiteSweepLeft_impl!(s, verbose=verbose)


function oneSiteSweepRight_impl!(s::OptimizeMPO; verbose::Int=1)
	kvals = Vector{Float64}()
	L = get_L(s)
	for j = L:-1:2
		(verbose > 2) && println("One site sweep from right to left on site $j.")
		distance = _compute_eff!(s, j, 'R')
		push!(kvals, distance)
		(verbose >= 2) && println("loss after updating: $distance.")
		_updateStorageRight!(s, j)
	end
	append!(get_kvals(s), kvals)
	return kvals
end

oneSiteSweepRight!(s::OptimizeMPO; verbose::Int=1) = oneSiteSweepRight_impl!(s, verbose=verbose)

function dosweep!(s::OptimizeMPO; verbose::Int=1)
	kvals1 = oneSiteSweepLeft!(s, verbose=verbose)
	kvals2 = oneSiteSweepRight!(s, verbose=verbose)
	append!(kvals1, kvals2)
	return kvals1
end

"""
	ql_compute!(dmrg; maxitr::Int=Constants.DEFAULT_QL_MAXITER, tol::QI_Real=Constants.DEFAULT_QL_STOP_TOL, verbose::Int=1, kwargs...)
	differ from dmrg_compute! by the way to compute the error.
"""
function ql_compute!(dmrg; maxitr::Int=50, tol::Float64=1.0e-6, verbose::Int=1, kwargs...)
	err = 1.
	i = 0
	converged = false
	while (i < maxitr)
		(verbose >= 2) && println("we are at the $i-th sweep...")
		Evals = dosweep!(dmrg; verbose=verbose, kwargs...)
		i += 1
		err = minimum(abs.(Evals))
		if (err < tol)
			converged = true
			(verbose > 2) && println("QL converges in $i sweeps.")
			break
		end
	end
	if !converged
		i = 0
		(verbose >= 1) && @warn "QL fails to converge in $maxitr sweeps."
	end
	return (i, err)
end

compute!(s::OptimizeMPO; maxitr::Int=50, tol::Real=1.0e-6, verbose::Int=1) = ql_compute!(s; maxitr=maxitr, tol=tol, verbose=verbose)

predict(s::OptimizeMPO, mps::NSymMPS) = get_mpo(s) * mps

_compute_y_norms(mpsys::Vector{<:NSymMPS}) = sum([real(dot(item, item)) for item in mpsys])

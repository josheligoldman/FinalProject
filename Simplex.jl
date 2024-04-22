using LinearAlgebra

function build_tableau(A::AbstractMatrix{T}, b::AbstractVector{T}, c::AbstractVector{T}, cons::S, vars::S) where {T<:AbstractFloat, S<:Integer}
	tvars = cons + vars

	tableau = zeros(T, cons + 2, tvars + 1)

	tableau[1:cons, 1:vars] = A
	tableau[1:cons, (vars+ 1):tvars] = Matrix{T}(I, cons, cons)
	tableau[1:cons, tvars + 1] = b
	tableau[cons + 1, 1:vars] = ones(T, 1, cons) * A
	tableau[cons + 1, tvars + 1] = (ones(T, 1, cons) * b)[1]
	tableau[cons + 2, 1:vars] = c

	col_to_basis = [(false, zero(S)) for _ in 1:tvars]
	for i in (vars + 1):tvars
		col_to_basis[i] = (true, i - vars)
	end
	row_to_basis = [i for i in (vars + 1):tvars]

	return (tableau, col_to_basis, row_to_basis)
end

function trim_tableau(tableau::AbstractMatrix{T}, cons::S, vars::S) where {S<:Integer, T<:AbstractFloat}
	tvars = cons + vars

	trimmed = zeros(T, cons + 1, vars + 1)
	trimmed[1:cons, 1:vars] = tableau[1:cons, 1:vars]
	trimmed[1:cons, vars + 1] = tableau[1:cons, tvars + 1]
	trimmed[cons + 1, 1:vars] = tableau[cons + 2, 1:vars]
	trimmed[cons + 1, vars + 1] = tableau[cons + 2, tvars + 1]

	return trimmed
end

function pivot!(A::AbstractMatrix{T}, row::S, col::S) where {T<:AbstractFloat, S<:Integer}
	if A[row, col] == 0
		error("Pivot element must be non-zero")
	end

	A[row, :] /= A[row, col]

	num_rows, _ = size(A)
	for i in 1:num_rows
		if i == row || A[i, col] == 0
			continue
		end
		A[i, :] -= A[row, :] * A[i, col] 
		A[i, col] = 0.0
	end
end

function min_ratio_test(tableau::AbstractMatrix{T}, cons::S, vars::S, col::S)::Integer where {T<:AbstractFloat, S<:Integer}
	min_ratio = Inf
	min_col = -1
	for i in 1:cons
		if tableau[i, col] <= 0
			continue
		end

		ratio = tableau[i, vars + 1] / tableau[i, col]
		if ratio < min_ratio
			min_ratio = ratio
			min_col = i
		end
	end
	return min_col
end

function bland!(tableau::AbstractMatrix{T}, col_to_basis::Vector{Tuple{Bool, S}}, row_to_basis::Vector{S}, cons::S, vars::S)::Bool where {T<:AbstractFloat, S<:Integer}
	enter_basis = -1
	for i in 1:vars
		if col_to_basis[i][1] || tableau[cons + 1, i] <= 0  # Skip if not basis or in basis but negative coefficient
			continue
		end

		enter_basis = i
		break
	end

	if enter_basis == -1
		# All coefficients are negative
		return true
	end

	exit_row = min_ratio_test(tableau, cons, vars, enter_basis)

	if exit_row == -1
		error("Column is entirely non-positive. Linear program is unbounded.")
	end

	exit_basis = row_to_basis[exit_row]

	pivot!(tableau, exit_row, enter_basis)

	col_to_basis[enter_basis] = (true, exit_row)
	col_to_basis[exit_basis] = (false, 0)

	row_to_basis[exit_row] = enter_basis

	return false
end

function simplex_solver!(tableau::AbstractMatrix{T}, col_to_basis::Vector{Tuple{Bool, S}}, row_to_basis::Vector{S}, cons::S, vars::S) where {T<:AbstractFloat, S<:Integer}
	#=
	Solves the standard form linear program 

	Maximize c^T x
	Subject to 
		Ax = b
		x >= 0
	=#
	counter = 0
	while !(bland!(tableau, col_to_basis, row_to_basis, cons, vars))
		counter += 1
	end

	return counter
end

function simplex_initializer(A::AbstractMatrix{T}, b::Vector{T}, c::Vector{T}, cons::S, vars::S) where {T<:AbstractFloat, S<:Integer}
	tvars = cons + vars

	tableau = zeros(T, cons + 2, tvars + 1)

	tableau[1:cons, 1:vars] = A
	tableau[1:cons, (vars+ 1):tvars] = Matrix{T}(I, cons, cons)
	tableau[1:cons, tvars + 1] = b
	tableau[cons + 1, 1:vars] = ones(T, 1, cons) * A
	tableau[cons + 1, tvars + 1] = (ones(T, 1, cons) * b)[1]
	tableau[cons + 2, 1:vars] = c
	
	col_to_basis = [(false, zero(S)) for _ in 1:tvars]
	for i in (vars + 1):tvars
		col_to_basis[i] = (true, i - vars)
	end
	row_to_basis = [i for i in (vars + 1):tvars]
	
	tableau, col_to_basis, row_to_basis = build_tableau(A, b, c, cons, vars)

	num_ops = simplex_solver!(tableau, col_to_basis, row_to_basis, cons, cons + vars)

	if tableau[cons + 1, tvars + 1] > 1e-6
		return (false, tableau, col_to_basis, row_to_basis, num_ops)
	end

	lo_ptr = 1
	hi_ptr = vars
	while hi_ptr < cons + vars
		hi_ptr += 1
		if !col_to_basis[hi_ptr][1]
			continue
		end

		while lo_ptr < vars && col_to_basis[lo_ptr][1]
			lo_ptr += 1
		end

		exit_row = col_to_basis[hi_ptr][2]
		# Careful: The pivot element might be 0. See page 16 of Goemans notes, Part 3. 
		pivot!(tableau, exit_row, lo_ptr)

		col_to_basis[lo_ptr] = (true, exit_row)
		col_to_basis[hi_ptr] = (false, 0)

		row_to_basis[exit_row] = lo_ptr
	end

	trimmed_tableau = trim_tableau(tableau, cons, vars)
	col_to_basis = col_to_basis[1:vars]

	return (true, trimmed_tableau, col_to_basis, row_to_basis, num_ops)
end

function simplex(A::AbstractMatrix{T}, b::Vector{T}, c::Vector{T}, cons::S, vars::S) where {S<:Integer, T<:AbstractFloat}
	flag, feasible_tableau, col_to_basis, row_to_basis, ops_init = simplex_initializer(A, b, c, cons, vars)

	if !flag
		error("Infeasible linear program.")
	end

	num_ops = simplex_solver!(feasible_tableau, col_to_basis, row_to_basis, cons, vars)

	optimal_val = -feasible_tableau[cons + 1, vars + 1]
	var_vals = T[0 for i in 1:vars]
	for i in 1:vars
		if !col_to_basis[i][1]
			continue
		end

		var_vals[i] = feasible_tableau[col_to_basis[i][2], vars + 1]
	end

	return (num_ops + ops_init, optimal_val, var_vals)
end

function tester() 
	cons = 3
	vars = 6

	A = Float64[1 0 0 1 0 0; 2 1 1 0 1 0; 2 2 1 0 0 1]
	b = Float64[4, 10, 16]
	c = Float64[20, 16, 12, 0, 0, 0]

	println("Input")
	display(A)
	display(b)
	display(c)

	num_ops, optimal_val, var_vals = simplex(A, b, c, cons, vars)

	println("Optimal Value ", optimal_val)
	println("Variable Values:")
	display(var_vals)
end

if abspath(PROGRAM_FILE) == @__FILE__
    tester()
end

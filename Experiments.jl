using Pkg
using Random
using Distributions
using LinearAlgebra
using Dates
using Plots
using ProgressBars
using LsqFit

include("Simplex.jl")

Random.seed!(Dates.datetime2epochms(now()))

function random_vector(distribution, length) 
	return rand(distribution, length)
end

function random_matrix(distribution, rows, cols)
	return rand(distribution, rows, cols)
end

function random_simplex(distribution, cons, vars, type=Float64)
	A = convert(
		Matrix{type},
		random_matrix(distribution, cons, vars)
	)
	x = convert(
		Vector{type},
		random_vector(distribution, vars)
	)
	b = A*x
	c = convert(
		Vector{type},
		random_vector(distribution, vars)
	)

	return (A, b, c)
end

function random_performance(distribution, cons, vars, type=Float64)
	A, b, c = random_simplex(distribution, cons, vars, type)

	try
		num_ops, _, _ = simplex(A, b, c, cons, vars)
		return num_ops
	catch e
		return -1
	end
end

function average_random_performance(samples, distribution, cons, vars, type=Float64)
	cnt = 0
	total = 0.0

	while cnt < samples
		ops = random_performance(distribution, cons, vars, type)
		if ops != -1
			cnt += 1
			total += ops
		else
			println("-1")
		end
	end

	return total / samples
end

function log_x_fit(x, y)
	log_x = log.(x)

	model(x, p) = p[1] * x .+ p[2]
	p0 = [0.0, 0.0]
	fit = curve_fit(model, log_x, y, p0)

	fitted_y = model(log_x, fit.param)

	plot!(x, fitted_y, label="Log X")

	println("Log X Params")
	println("Slope (m): " * string(fit.param[1]))
	println("Intercept (b): " * string(fit.param[2]))
end

function exp_fit(x, y)
	log_y = log.(y)

	model(x, p) = p[1] * x .+ p[2]
	p0 = [0.0, 0.0]
	fit = curve_fit(model, x, log_y, p0)

	fitted_y = exp.(model(x, fit.param))

	plot!(x, fitted_y, label="Exp")

	println("Exp Params")
	println("Slope (m): " * string(fit.param[1]))
	println("Intercept (b): " * string(fit.param[2]))
end

function power_law_fit(x, y)
	log_x = log.(x)
	log_y = log.(y)

	model(x, p) = p[1] * x .+ p[2]
	p0 = [0.0, 0.0]
	fit = curve_fit(model, log_x, log_y, p0)

	fitted_y = exp.(model(log_x, fit.param))
	plot!(x, fitted_y, label="Power Law")

	println("Power Law Params")
	println("Slope (m): " * string(fit.param[1]))
	println("Intercept (b): " * string(fit.param[2]))
end

function linear_fit(x, y)
	model(x, p) = p[1] * x .+ p[2]
	p0 = [0.0, 0.0]
	fit = curve_fit(model, x, y, p0)

	fitted_y = model(x, fit.param)
	plot!(x, fitted_y, label="Linear")

	println("Linear Params")
	println("Slope (m): " * string(fit.param[1]))
	println("Intercept (b): " * string(fit.param[2]))
end

function vars_increase(distribution; cons=10, samples=20)
	x = cons:(cons + 500)

	y = zeros(0)
	for i in ProgressBar(x)
		append!(y, average_random_performance(samples, distribution, cons, i))
	end

	plot(x, y, title="Pivot Operations vs Variables", label="Data", xlabel="Variables", ylabel="Operations")

	log_x_fit(x, y)
	power_law_fit(x, y)

	savefig(string("Vars_" * string(cons) * ".png"))
end

function cons_and_vars_increase(distribution)
	samples = 20

	x = 10:100

	y = zeros(0)
	for i in ProgressBar(x)
		append!(y, average_random_performance(samples, distribution, i, 2*i))
	end

	plot(
		x, 
		y, 
		title="Pivot Operations vs Size", 
		label="Data", 
		xlabel="Size", 
		ylabel="Operations"
	)

	power_law_fit(x, y)
	exp_fit(x, y)

	savefig("Cons_Vars.png")
end

function perturbation(distribution, cons, vars; is_A=true, is_b=true, is_c=true)
	A, b, c = random_simplex(distribution, cons, vars)

	norm = Normal(0, 1)

	noise_A = random_matrix(norm, cons, vars)
	noise_b = random_vector(norm, cons)
	noise_c = random_vector(norm, vars)

	pA = A + is_A * noise_A
	pb = b + is_b * noise_b
	pc = c + is_c * noise_c

	try
		_, optim, _ = simplex(A, b, c, cons, vars)
		_, p_optim, _ = simplex(pA, pb, pc, cons, vars)

		return abs(optim - p_optim)
	catch e
		return -1
	end

	return -1
end

function average_perturbation(distribution, cons, vars; samples=20, is_A=false, is_b=false, is_c=false)
	avg = 0.0
	cnt = 0
	while cnt < samples
		pert = perturbation(distribution, cons, vars; is_A=true)
		if pert != -1
			avg += pert
			cnt += 1
		end
	end

	return avg / samples
end

function range_perturbation(distribution; samples=50, is_A=false, is_b=false, is_c=false)
	x = 10:40

	y = zeros(0)
	for i in ProgressBar(x)
		append!(y, average_perturbation(distribution, i, i + 10; samples=samples, is_A=is_A, is_b=is_b, is_c=is_c))
	end
	
	plot(
		x,
		y,
		title="Perturbation vs Size",
		label="Data",
		xlabel="Size",
		ylabel="Perturbation"
	)

	linear_fit(x, y)

	savefig("Perturbation_" * "A" * string(Int(is_A)) * "b" * string(Int(is_b)) * "c" * string(Int(is_c)) * ".png")
end


function main() 
	println("Experiments:")

	dist1 = DiscreteUniform(0, 10)

	# vars_increase(dist1; cons=30, samples=20)
	cons_and_vars_increase(dist1)

	# range_perturbation(dist1; is_c=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

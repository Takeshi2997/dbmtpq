include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

# Make data file
dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir(dirname)

f = open("error.txt", "w")
for iϵ in 1:1

    ϵ = 0.06

    filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"

    # Initialize weight, bias and η
    weight1 = Init.weight(Const.dimB, Const.dimM)
    w1moment = zeros(Float32, Const.dimB, Const.dimM)
    w1velocity = zeros(Float32, Const.dimB, Const.dimM)
    weight2 = Init.weight(Const.dimM, Const.dimS)
    w2moment = zeros(Float32, Const.dimM, Const.dimS)
    w2velocity = zeros(Float32, Const.dimM, Const.dimS)
    biasB = Init.bias(Const.dimB)
    bmomentB = zeros(Float32, Const.dimB)
    bvelocityB = zeros(Float32, Const.dimB)
    biasM = Init.bias(Const.dimM)
    bmomentM = zeros(Float32, Const.dimM)
    bvelocityM = zeros(Float32, Const.dimM)
    biasS = Init.bias(Const.dimS)
    bmomentS = zeros(Float32, Const.dimS)
    bvelocityS = zeros(Float32, Const.dimS)
    η = 0.9
    error = 0.0
    energyS = 0.0
    energyB = 0.0

    # Define network
    network = (weight1, weight2, biasB, biasM, biasS, η)
    
    # Learning
    for it in 1:Const.it_num
        error, energy, energyS, energyB, dispersion, dweight1, dweight2,
        dbiasB, dbiasM, dbiasS = MLcore.diff_error(network, ϵ)

        # Adam
        lr_t = Const.lr * sqrt(1.0 - 0.999^it) / (1.0 - 0.9^it)
        w1moment += (1 - 0.9) * (dweight1 - w1moment)
        w1velocity += (1 - 0.999) * (dweight1.^2 - w1velocity)
        weight1 -= lr_t * w1moment ./ (sqrt.(w1velocity) .+ 1.0 * 10^(-7))
        w2moment += (1 - 0.9) * (dweight2 - w2moment)
        w2velocity += (1 - 0.999) * (dweight2.^2 - w2velocity)
        weight2 -= lr_t * w2moment ./ (sqrt.(w2velocity) .+ 1.0 * 10^(-7))
        bmomentB += (1 - 0.9) * (dbiasB - bmomentB)
        bvelocityB += (1 - 0.999) * (dbiasB.^2 - bvelocityB)
        biasB -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
        bmomentM += (1 - 0.9) * (dbiasM - bmomentM)
        bvelocityM += (1 - 0.999) * (dbiasM.^2 - bvelocityM)
        biasM -= lr_t * bmomentM ./ (sqrt.(bvelocityM) .+ 1.0 * 10^(-7))
        bmomentS += (1 - 0.9) * (dbiasS - bmomentS)
        bvelocityS += (1 - 0.999) * (dbiasS.^2 - bvelocityS)
        biasS -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))

        write(f, string(it))
        write(f, "\t")
        write(f, string(error))
        write(f, "\t")
        write(f, string(dispersion))
        write(f, "\t")
        write(f, string(energyS))
        write(f, "\t")
        write(f, string(energyB))
        write(f, "\n")
    
        network = (weight1, weight2, biasB, biasM, biasS, η)
    end

    # Write error
#    write(f, string(iϵ))
#    write(f, "\t")
#    write(f, string(error))
#    write(f, "\t")
#    write(f, string(energyS))
#    write(f, "\t")
#    write(f, string(energyB))
#    write(f, "\n")
    
    open(io -> serialize(io, network), filename, "w")
end
close(f)

module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(network, ϵ)

        (weight1, weight2, biasB, biasM, biasS, η) = network
        n = rand([0.0, 1.0], Const.dimB)
        energy = 0.0
        energyS = 0.0
        energyB = 0.0
        squareenergy = 0.0
        dweight1_h2 = zeros(Float32, Const.dimB, Const.dimM)
        dweight1_h = zeros(Float32, Const.dimB, Const.dimM)
        dweight1 = zeros(Float32, Const.dimB, Const.dimM)
        dweight2_h2 = zeros(Float32, Const.dimM, Const.dimS)
        dweight2_h = zeros(Float32, Const.dimM, Const.dimS)
        dweight2 = zeros(Float32, Const.dimM, Const.dimS)
        dbiasB_h2 = zeros(Float32, Const.dimB)
        dbiasB_h = zeros(Float32, Const.dimB)
        dbiasB = zeros(Float32, Const.dimB)
        dbiasM_h2 = zeros(Float32, Const.dimM)
        dbiasM_h = zeros(Float32, Const.dimM)
        dbiasM = zeros(Float32, Const.dimM)
        dbiasS_h2 = zeros(Float32, Const.dimS)
        dbiasS_h = zeros(Float32, Const.dimS)
        dbiasS = zeros(Float32, Const.dimS)

        for i in 1:Const.iters_num+Const.burnintime
            activationB = transpose(n) * weight1 .+ biasM
            x = Func.updateM(activationB)
            activationM1 = transpose(x) * weight2 .+ biasS
            s = Func.updateS(activationM1)
            activationM2 = weight2 * s .+ biasM
            x = Func.updateM(activationM2)
            activationB = weight1 * x + biasB
            n = Func.updateB(activationB)
            if i > Const.burnintime
                e = Func.hamiltonian(n, s)
                e2 = Func.squarehamiltonian(n, s)
                energy += e
                energyS += Func.energyS(s)
                energyB += Func.energyB(n)
                squareenergy += e2
                dweight1_h2 +=  transpose(x) .* n .* e2
                dweight1_h +=  transpose(x) .* n .* e
                dweight1 +=  transpose(x) .* n
                dweight2_h2 +=  transpose(s) .* x .* e2
                dweight2_h +=  transpose(s) .* x .* e
                dweight2 +=  transpose(s) .* x
                dbiasB_h2 += n * e2
                dbiasB_h += n * e
                dbiasB += n
                dbiasM_h2 += x * e2
                dbiasM_h += x * e
                dbiasM += x
                dbiasS_h2 += s * e2
                dbiasS_h += s * e
                dbiasS += s
            end
        end
        energy /= Const.iters_num
        energyS /= Const.iters_num
        energyB /= Const.iters_num
        squareenergy /= Const.iters_num
        dweight1_h2 /= Const.iters_num
        dweight1_h /= Const.iters_num
        dweight1 /= Const.iters_num
        dweight2_h2 /= Const.iters_num
        dweight2_h /= Const.iters_num
        dweight2 /= Const.iters_num
        dbiasB_h2 /= Const.iters_num
        dbiasB_h /= Const.iters_num
        dbiasB /= Const.iters_num
        dbiasM_h2 /= Const.iters_num
        dbiasM_h /= Const.iters_num
        dbiasM /= Const.iters_num
        dbiasS_h2 /= Const.iters_num
        dbiasS_h /= Const.iters_num
        dbiasS /= Const.iters_num
        dispersion = squareenergy - energy^2
        error = η * (energy - ϵ)^2 + dispersion

        diff_weight1 = (dweight1_h2 - squareenergy * dweight1_h) + 
        2.0 * ((η - 1.0) * energy - ϵ * η) * (dweight1_h - energy * dweight1)
        diff_weight2 = (dweight2_h2 - squareenergy * dweight2_h) + 
        2.0 * ((η - 1.0) * energy - ϵ * η) * (dweight2_h - energy * dweight2)
        diff_biasB = (dbiasB_h2 - squareenergy * dbiasB_h) + 
        2.0 * ((η - 1.0) * energy - ϵ * η) * (dbiasB_h - energy * dbiasB)
        diff_biasM = (dbiasM_h2 - squareenergy * dbiasM_h) + 
        2.0 * ((η - 1.0) * energy - ϵ * η) * (dbiasM_h - energy * dbiasM)
        diff_biasS = (dbiasS_h2 - squareenergy * dbiasS_h) + 
        2.0 * ((η - 1.0) * energy - ϵ * η) * (dbiasS_h - energy * dbiasS)

        return error, energy, energyS, energyB, dispersion, 
        diff_weight1, diff_weight2, diff_biasB, diff_biasM, diff_biasS
    end

    function forward(network)

        (weight, biasB, biasS, η) = network
        s = [1.0, 1.0]
        energyS = 0.0
        energyB = 0.0
        num = 10000

        for i in 1:num+Const.burnintime
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            if i > Const.burnintime
                energyS += Func.energyS(s)
                energyB += Func.energyB(n)
            end
        end
        energyS /= num
        energyB /= num

        return energyS, energyB
    end
end

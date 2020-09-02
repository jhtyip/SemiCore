######################### Functions for solving system of linear equations #########################
function gaussian_elimination!(A::Array{Float64,2})
    rows = size(A, 1)
    cols = size(A, 2)

    row = 1
    for col = 1:(cols - 1)
        max_index = argmax(abs.(A[row:end, col])) + row - 1

        if (A[max_index, col] == 0)
            println("matrix is singular!")
            continue
        end

        temp_vector = A[max_index, :]
        A[max_index, :] = A[row, :]
        A[row, :] = temp_vector

        for i = (row + 1):rows
            fraction = A[i, col] / A[row, col]
            for j = (col + 1):cols
                 A[i, j] -= A[row, j] * fraction
            end
            A[i, col] = 0
        end

        row += 1
    end
end


function back_substitution(A::Array{Float64,2})
    rows = size(A, 1)
    cols = size(A, 2)

    soln = zeros(rows)
    for i = rows:-1:1
        sum = 0.0
        for j = rows:-1:i
            sum += soln[j] * A[i, j]
        end
        soln[i] = (A[i, cols] - sum) / A[i, i]
    end

    return soln
end


function gauss_jordan_elimination!(A::Array{Float64,2})
    rows = size(A, 1)
    cols = size(A, 2)

    row = 1
    for col = 1:cols - 1
        if (A[row, col] != 0)
            for i = cols:-1:col
                A[row, i] /= A[row, col]
            end

            for i = 1:row - 1
                for j = cols:-1:col
                    A[i, j] -= A[i, col] * A[row, j]
                end
            end

            row += 1
        end
    end
end


######################################### For dmOnly() #############################################
# Return mass in a shell according to the NFW profile
function NFW_shellMass(NFW_params, shellRange)
    # NFW_params = [rho_0, R_s, c]
    # shellRange = [r_1, r_2] where r_1 < r_2

    result_integrand(r) = 4 * pi * NFW_params[1] * NFW_params[2] ^ 3 * (NFW_params[2] / (NFW_params[2] + r) + log(NFW_params[2] + r))

    return result_integrand(shellRange[2]) - result_integrand(shellRange[1])
end


# Return mass array of NFW profile
# shells_radii[i] = [inner radius, outer radius, shell radius] in the ith row where shell radius = (inner + outer) / 2
function NFW_shells(NFW_params, numOfShells, shellThicknessFactor, extend_factor)    
    NFW_R_max = NFW_params[2] * NFW_params[3] * extend_factor

    # Exponentially increasing shellThickness
    firstShellThickness = NFW_R_max * (1 - shellThicknessFactor) / (1 - shellThicknessFactor ^ numOfShells)

    shells_radii = zeros(numOfShells, 3)
    shells_mass = zeros(size(shells_radii, 1))
    for i in 1:size(shells_radii, 1)
        shells_radii[i, 1] = firstShellThickness * (1 - shellThicknessFactor ^ (i - 1)) / (1 - shellThicknessFactor)  # Inner radius
        shells_radii[i, 2] = shells_radii[i, 1] + firstShellThickness * shellThicknessFactor ^ (i - 1)  # Outer radius
        shells_radii[i, 3] = (shells_radii[i, 1] + shells_radii[i, 2]) / 2  # Shell radius
        shells_mass[i] = NFW_shellMass(NFW_params, shells_radii[i, 1:2])
    end

    return shells_radii, shells_mass
end


# Return enclosed mass array from mass array
function enclosedMass(shells_radii, shells_mass)
    shells_enclosedMass = zeros(size(shells_radii, 1))
    for i in 1:size(shells_enclosedMass, 1)
        shells_enclosedMass[i] = sum(shells_mass[1:i])
    end

    return shells_enclosedMass
end


# Return GPE (per mass) array of the NFW profile (analytic)
function NFW_GPE(NFWshells_radii, NFW_params, G)
    NFWshells_GPE = zeros(size(NFWshells_radii, 1))
    for i in 1:size(NFWshells_GPE, 1)
        NFWshells_GPE[i] = -4 * pi * G * NFW_params[1] * NFW_params[2] ^ 3 / NFWshells_radii[i, 3] * log(1 + NFWshells_radii[i, 3] / NFW_params[2])
    end

    return NFWshells_GPE
end


# Return GPE (per mass) array from a mass array (numerical)
function GPE(shells_radii, shells_mass, shells_enclosedMass, G)
    shells_GPE = zeros(size(shells_radii, 1))
    
    for i in 1:size(shells_radii, 1)
        shells_GPE[i] = -G * shells_enclosedMass[i] / shells_radii[i, 3]

        if i < size(shells_radii, 1)
            GPEbyOuterShells = 0
            for j in i + 1:size(shells_radii,1)
                GPEbyOuterShells += -G * shells_mass[j] / shells_radii[j, 3]
            end            
            shells_GPE[i] += GPEbyOuterShells
        end
    end

    return shells_GPE
end


# Return angular momentum (per mass) array
function L(shells_radii, shells_enclosedMass, G)
    shells_L = zeros(size(shells_radii, 1))
    for i in 1:size(shells_L, 1)
        shells_L[i] = (G * shells_enclosedMass[i] * shells_radii[i, 3]) ^ (1 / 2)
    end

    return shells_L
end


# Return total energy (per mass) array of daughters just-born at different radii (radii of the circular orbits of their mothers)
function totalE_afterDecay(shells_radii, shells_GPE, shells_L, v_k)
    shells_totalE_afterDecay = zeros(size(shells_radii, 1))
    for i in 1:size(shells_totalE_afterDecay, 1)
        shells_totalE_afterDecay[i] = shells_GPE[i] + (shells_L[i] / shells_radii[i, 3]) ^ 2 / 2 + v_k ^ 2 / 2
    end

    return shells_totalE_afterDecay
end


# Return value of U_eff(r) - E_dau for different r
function energyEquation(r, L, totalE_afterDecay, Tshells_radii, Tshells_GPE)
    if r <= 0  # Rejected
        return zeros(NaN)  # To cause error, halting the program
    elseif r <= Tshells_radii[1, 3]  # r small
        return Tshells_GPE[1] + (L / r) ^ 2 / 2 - totalE_afterDecay
    elseif r > Tshells_radii[end, 3]
        return  Tshells_GPE[end] * Tshells_radii[end, 3] / r + (L / r) ^ 2 / 2 - totalE_afterDecay  
    else  # r in between: value by interpolation
        radiusIndex = -1
        for i in 2:size(Tshells_radii, 1)
            if r <= Tshells_radii[i, 3]
                radiusIndex = i
                break
            end
        end
        intervalSlope = (Tshells_GPE[radiusIndex] - Tshells_GPE[radiusIndex - 1]) / (Tshells_radii[radiusIndex, 3] - Tshells_radii[radiusIndex - 1, 3])
        intervalIntercept = Tshells_GPE[radiusIndex] - intervalSlope * Tshells_radii[radiusIndex, 3]
        radiusGPE = intervalSlope * r + intervalIntercept

        return radiusGPE + (L / r) ^ 2 / 2 - totalE_afterDecay
    end
end


# Return r_min, r_max of the daughter's orbit
# Search in [l1, l2] U [r1, r2] using the bisection method
function ellipseSolver(r_0, L, totalE_afterDecay, shells_radii, Tshells_GPE, tol_ellipseGuess)
    firstShellThickness = shells_radii[1, 2]  # To be used as a part of the tolerance for the bisection method
   
    # Some initial checking
    if energyEquation(r_0, L, totalE_afterDecay, shells_radii, Tshells_GPE) >= 0
        # This should not happen unless GPE/totalE are not updated properly (= 0 occurs when v_k = 0)
        println("ellipseSolver: v_k probably too small; no solvable roots")
        
        # println(energyEquation(r_0, L, totalE_afterDecay, Tshells_radii, Tshells_GPE, Tshells_enclosedMass))
        # zeros(NaN)  # To cause error, halting the program
        return r_0, r_0  # If this happens, radii just stay put (i.e. solution for v_k = 0)
    elseif totalE_afterDecay >= 0  # Escaped
        return -1, -1
    else  # If checking passed
        l2 = r_0
        r1 = r_0
    end

    # Setting l1 and r2
    l1 = firstShellThickness
    while energyEquation(l1, L, totalE_afterDecay, shells_radii, Tshells_GPE) <= 0
        l1 /= 2
    end
    r2 = shells_radii[end, 3]
    while energyEquation(r2, L, totalE_afterDecay, shells_radii, Tshells_GPE) <= 0
        r2 *= 2
    end

    # Bisection method
    lastDiff = 0
    while (l2 - l1 > firstShellThickness * tol_ellipseGuess) && (l2 - l1 != lastDiff)
        lastDiff = l2 - l1
        l3 = (l1 + l2) / 2
        energyEquation_value = energyEquation(l3, L, totalE_afterDecay, shells_radii, Tshells_GPE)
        if energyEquation_value < 0
            l2 = l3
        elseif energyEquation_value > 0
            l1 = l3
        else
            l1 = l3
            l2 = l3
        end
    end
    lastDiff = 0
    while (r2 - r1 > firstShellThickness * tol_ellipseGuess) && (r2 - r1 != lastDiff)
        lastDiff = r2 - r1
        r3 = (r2 + r1) / 2
        energyEquation_value = energyEquation(r3, L, totalE_afterDecay, shells_radii, Tshells_GPE)
        if energyEquation_value < 0
            r1 = r3
        elseif energyEquation_value > 0
            r2 = r3
        else
            r1 = r3
            r2 = r3
        end
    end

    root1 = (l1 + l2) / 2
    root2 = (r1 + r2) / 2
    return root1, root2
end


# Return array of r_min and r_max at each radius
function ellipseRadii(shells_L, shells_totalE_afterDecay, Tshells_radii, Tshells_GPE, tol_ellipseGuess)
    shells_ellipseRadii = zeros(size(Tshells_radii, 1), 2)

    for i in 1:size(shells_ellipseRadii, 1)
        root1, root2 = ellipseSolver(Tshells_radii[i, 3], shells_L[i], shells_totalE_afterDecay[i], Tshells_radii, Tshells_GPE, tol_ellipseGuess)

        shells_ellipseRadii[i, 1] = root1
        shells_ellipseRadii[i, 2] = root2
    end

    return shells_ellipseRadii
end


# Return weightFactor (g-function) array for a particular r_ref where r_ref specifies the enclosed region being concerned
function weightFactorArray(r_ref, shells_ellipseRadii, L, shells_totalE, Tshells_GPE, Tshell_radii, Tshells_enclosedMass, t_i, orderOfpolynomial, G, NFW_params)
    weightFactor = zeros(size(shells_ellipseRadii, 1))
   
    for i in 1:size(weightFactor, 1)  # Looping each r_0
        r_max = shells_ellipseRadii[i, 2]
        r_min = shells_ellipseRadii[i, 1]

        if r_max == -1 && r_min == -1  # Escaped the whole system
            weightFactor[i] = 0
        elseif r_min > r_ref
            weightFactor[i] = 0
        elseif r_max <= r_ref
            weightFactor[i] = 1
        else
            weightFactor[i] = weightFactorSolver(r_ref, r_max, r_min, L[i], shells_totalE[i], Tshells_GPE, Tshell_radii, Tshells_enclosedMass, t_i, orderOfpolynomial, G, NFW_params)
        end
    end
    
    return weightFactor
end


# For weightFactorSolver()
function U_eff_gfunction(x, r_max, r_min, L, Tshells_radii, Tshells_GPE)
    r = x * (r_max - r_min) + r_min

    if r <= 0  # Rejected
        return zeros(NaN)  # To cause error, halting the program
    elseif r <= Tshells_radii[1, 3]  # r small
        return Tshells_GPE[1]  + (L / r) ^ 2 / 2 
    elseif r > Tshells_radii[end, 3]  # r big
        return Tshells_GPE[end] * Tshells_radii[end, 3] / r + (L / r) ^ 2 / 2 
    else  # r in between: value by interpolation
        radiusIndex = -1
        for i in 2:size(Tshells_radii, 1)
            if r <= Tshells_radii[i, 3]
                radiusIndex = i
                break
            end
        end
        intervalSlope = (Tshells_GPE[radiusIndex] - Tshells_GPE[radiusIndex - 1]) / (Tshells_radii[radiusIndex, 3] - Tshells_radii[radiusIndex - 1, 3])
        intervalIntercept = Tshells_GPE[radiusIndex] - intervalSlope * Tshells_radii[radiusIndex, 3]
        radiusGPE = intervalSlope * r + intervalIntercept

        return radiusGPE + (L / r) ^ 2 / 2 
    end
end


# For weightFactorSolver()
function dU_eff(r, L, Tshells_radii, Tshells_enclosedMass)
    if r <= 0  # Rejected
        println("dU_eff: dead end")
        return zeros(NaN)  # To cause error, halting the program
    elseif r <= Tshells_radii[1, 3]  # r small
        return G * Tshells_enclosedMass[1] / Tshells_radii[1, 3] / r - L ^ 2 / r ^ 3
    elseif r > Tshells_radii[end, 3]  # r big
        return G * Tshells_enclosedMass[end] / r ^ 2 - L ^ 2 / r ^ 3
    else  # r in between: value by interpolation
        radiusIndex = -1
        for i in 2:size(Tshells_radii, 1)
            if r <= Tshells_radii[i, 3]
                radiusIndex = i
                break
            end
        end
        intervalSlope = (Tshells_enclosedMass[radiusIndex] - Tshells_enclosedMass[radiusIndex - 1]) / (Tshells_radii[radiusIndex, 3] - Tshells_radii[radiusIndex - 1, 3])
        intervalIntercept = Tshells_enclosedMass[radiusIndex] - intervalSlope * Tshells_radii[radiusIndex, 3]
        radiusMass = intervalSlope * r + intervalIntercept

        return G * radiusMass / r ^ 2  - L ^ 2 / r ^ 3
    end
end


# For weightFactorSolver()
function dU_eff_NFW(r, NFW_params, G, L)
    rho_0 = NFW_params[1]
    R_s = NFW_params[2]

    dUdr = 4 * pi * G * rho_0 * R_s ^ 3 / r ^ 2 * log(1 + r / R_s) - 4 * pi * G * rho_0 * R_s ^ 2 / r / (1 + r / R_s)   

    return dUdr - L ^ 2 /  r ^ 3
end


# Evaluating g-function
function weightFactorSolver(r_ref, r_max, r_min, L, E, Tshells_GPE, Tshells_radii, Tshells_enclosedMass, t_i, orderOfpolynomial, G, NFW_params)
    U_eff(x) = U_eff_gfunction(x, r_max, r_min, L, Tshells_radii, Tshells_GPE)
    
    # Normalizing r
    x_min = 0
    x_max = 1
    x_ref = (r_ref - r_min) / (r_max - r_min)
    
    if t_i == 2  # Assume NFW halo at initialization
        dU_max = dU_eff_NFW(r_max, NFW_params, G, L) * (r_max - r_min)
        dU_min = dU_eff_NFW(r_min, NFW_params, G, L) * (r_max - r_min)
    else
        dU_max = dU_eff(r_max, L, Tshells_radii, Tshells_enclosedMass) * (r_max - r_min)
        dU_min = dU_eff(r_min, L, Tshells_radii, Tshells_enclosedMass) * (r_max - r_min)
    end

    res(x) = 1 / sqrt(E - U_eff(x)) - 1 / sqrt(-1 * dU_min * (x - x_min)) - 1 / sqrt(dU_max * (x_max - x))  # Residue function
    
    x_a = 0.5  # Reference point of Taylor series expansion of res(x)

    # Set up linear equations for determining the polynomial approximation of res(x)
    # res(x) = a + b x + c x^2 + d x^3 + ... up to x^n order. To solve for coefficients, n + 1 points are required 
    Number_of_point = orderOfpolynomial + 1
    dx = 0.998 / (Number_of_point - 1)
    equation_matrix = zeros(Number_of_point, Number_of_point + 1)
    for i in 1:Number_of_point
        # Choose x_i for i-th linear equation 
        # Avoid choosing x_i at the boundaries of its domain (to avoid singularities)
        x =  0.001 + (i - 1) * dx
        for j in 1:Number_of_point
            equation_matrix[i, j] = (x - x_a) ^ (j - 1)
        end
        equation_matrix[i, end] = res(x)
    end

    soln = zeros(Number_of_point)  # Solution array for the linear equations
    # Solve matrix by Gaussian elimination 
    gaussian_elimination!(equation_matrix)
    gauss_jordan_elimination!(equation_matrix)
    soln = back_substitution(equation_matrix)
    
    I_res_max = I_res_min = I_res_ref = 0 
    for i in 1:Number_of_point
        I_res_max += soln[i] * (x_max - x_a) ^ i / i
        I_res_min += soln[i] * (x_min - x_a) ^ i / i
        I_res_ref += soln[i] * (x_ref - x_a) ^ i / i
    end    
    
    # N1, N2 (D1, D2) are analytic integration results of the last two expressions in the modified integrand in the nominator (denominator) of the g-function
    N1 = 2 * sqrt((x_ref - x_min) / (-dU_min))
    N2 = 2 * (sqrt(x_max - x_min) - sqrt(x_max - x_ref)) / sqrt(dU_max)
    D1 = 2 * sqrt((x_max - x_min) / (-dU_min))
    D2 = 2 * sqrt((x_max - x_min) / (dU_max))

    nominator = I_res_ref - I_res_min + N1 + N2
    denominator = I_res_max - I_res_min + D1 + D2
    
    return nominator / denominator
end


# Decaying mothers and distributing daughters
function updateShellsMass(shells_radii, shells_ellipseRadii, Mshells_mass, p_undecayed, L, shells_totalE, Tshells_GPE, Tshells_enclosedMass, t_i, orderOfpolynomial, G, NFW_params)
    Mshells_decayedMass = Mshells_mass * (1 - p_undecayed)  # Daughters to be redistributed
    Mshells_mass *= p_undecayed  # Remaining mothers

    Dshells_enclosedMass_decayedMass = zeros(size(shells_radii, 1))
    for i in 1:size(Dshells_enclosedMass_decayedMass, 1)
        weightFactor = weightFactorArray(shells_radii[i, 2], shells_ellipseRadii, L, shells_totalE, Tshells_GPE, shells_radii, Tshells_enclosedMass, t_i, orderOfpolynomial, G, NFW_params)
        Dshells_enclosedMass_decayedMass[i] = sum(Mshells_decayedMass .* weightFactor)
    end

    Dshells_decayedMass = zeros(size(Dshells_enclosedMass_decayedMass, 1))
    if Dshells_decayedMass != []  # If not all daughters at all radii escape
        Dshells_decayedMass[1] = Dshells_enclosedMass_decayedMass[1]
        for i in 2:size(Dshells_decayedMass, 1)
            Dshells_decayedMass[i] = Dshells_enclosedMass_decayedMass[i] - Dshells_enclosedMass_decayedMass[i - 1]
        end
    end

    return Mshells_mass, Dshells_decayedMass
end


# Return mass array after adiabatic expansion
function adiabaticExpansion(shells_radii, shells_mass, Tshells_enclosedMass, Tshells_enclosedMass_updated) 
    expansionRatios = Tshells_enclosedMass[1:size(shells_radii, 1)] ./ Tshells_enclosedMass_updated[1:size(shells_radii, 1)]

    # To check whether there are contractions instead of only expansions (doesn't really matter)
    contractionCount = count(i -> (i < 1), expansionRatios)
    if contractionCount > 0
        # println("adiabaticExpansion: expansion ratio smaller than 1, i.e. NOT expanding. Count: ", contractionCount, ", min ratio: ", findmin(expansionRatios)[1])
        # zeros(NaN)  # To cause error, halting the program
    end

    # shells_expandedRadii = shells_radii[:, 3] .* expansionRatios
    shells_expandedRadii = shells_radii[:, 2] .* expansionRatios  # Use outer or shell radii for the set of points for interpolation? outer ([:, 2]) is recommended

    # To make sure expandedRadii is "monotonic"
    violationCount = 0
    checkedEntry = -1
    while checkedEntry != size(shells_expandedRadii, 1) - 1
        checkedEntry = -1
        for i in 1:size(shells_expandedRadii, 1) - 1
            if shells_expandedRadii[i] > shells_expandedRadii[i + 1]
                violationCount += 1

                eR_1 = shells_expandedRadii[i]
                eR_2 = shells_expandedRadii[i + 1]
                shells_expandedRadii[i] = eR_2
                shells_expandedRadii[i + 1] = eR_1

                break
            else
                checkedEntry = i
            end
        end
    end
    if violationCount > 0
        println("adiabaticExpansion: violationCount = ", violationCount)
    end

    expandedShells_radii = shells_radii
    expandedShells_mass = zeros(size(shells_radii, 1), 1)
    for i in 1:size(shells_radii, 1)  # This interpolation method should work if the relation between old and expanded radii is monotonic. Check total mass after expansion
        e1 = shells_radii[i, 1]  # Inner radius of expanded shells
        e2 = shells_radii[i, 2]  # Outer radius of expanded shells
        
        e1_smallerThanID = -1
        for j in 1:size(shells_expandedRadii, 1)
            if e1 < shells_expandedRadii[j]
                e1_smallerThanID = j
                break
            end
        end
        
        e2_smallerThanID = -1
        for j in 1:size(shells_expandedRadii, 1)
            if e2 < shells_expandedRadii[j]
                e2_smallerThanID = j
                break
            end
        end
        
        if e1_smallerThanID == 1
            m = (shells_radii[e1_smallerThanID, 2] - 0) / (shells_expandedRadii[e1_smallerThanID] - 0)
            c = 0
            r1 = m * e1 + c
        elseif e1_smallerThanID != -1
            m = (shells_radii[e1_smallerThanID, 2] - shells_radii[e1_smallerThanID - 1, 2]) / (shells_expandedRadii[e1_smallerThanID] - shells_expandedRadii[e1_smallerThanID - 1])
            c = shells_radii[e1_smallerThanID, 2] - m * shells_expandedRadii[e1_smallerThanID]
            r1 = m * e1 + c
        else
            r1 = -1  # Should never happen
        end
        
        if e2_smallerThanID == 1
            m = (shells_radii[e2_smallerThanID, 2] - 0) / (shells_expandedRadii[e2_smallerThanID] - 0)
            c = 0
            r2 = m * e2 + c
        elseif e2_smallerThanID != -1
            m = (shells_radii[e2_smallerThanID, 2] - shells_radii[e2_smallerThanID - 1, 2]) / (shells_expandedRadii[e2_smallerThanID] - shells_expandedRadii[e2_smallerThanID - 1])
            c = shells_radii[e2_smallerThanID, 2] - m * shells_expandedRadii[e2_smallerThanID]
            r2 = m * e2 + c
        else
            r2 = -1  # Will happen once
            # println("adiabaticExpansion: r2 = -1")
        end
        
        firstShellThickness = shells_radii[1, 2]
        shellThicknessFactor = (shells_radii[2, 2] - shells_radii[2, 1]) / firstShellThickness
        if r1 != -1
            totalLen = 0
            r1_smallerThanID = 0
            while totalLen <= r1
                r1_smallerThanID += 1
                totalLen += firstShellThickness * shellThicknessFactor ^ (r1_smallerThanID - 1)
            end
            
            if r1_smallerThanID > size(shells_radii, 1)
                println("adiabatic Expansion error: r1 > outermost radius")  # Prompt error
                continue  # Hotfix for weird boundary cases
            end
        else
            println("adiabaticExpansion error: r1 = -1")  # Prompt error
            continue  # Hotfix for weird boundary cases
        end
        
        if r2 != -1
            totalLen = 0
            r2_smallerThanID = 0
            while totalLen <= r2
                r2_smallerThanID += 1
                totalLen += firstShellThickness * shellThicknessFactor ^ (r2_smallerThanID - 1)
            end
        else
            r2_smallerThanID = -1  # Special treatment
        end
       
        expandedShells_mass[i] += shells_mass[r1_smallerThanID] * (1 - (r1 ^ 3 - shells_radii[r1_smallerThanID, 1] ^ 3) / (shells_radii[r1_smallerThanID, 2] ^ 3 - shells_radii[r1_smallerThanID, 1] ^ 3))
        if r2_smallerThanID == -1
            expandedShells_mass[i] += shells_mass[end]
            r2_smallerThanID = size(shells_radii, 1)
        else
            expandedShells_mass[i] += shells_mass[r2_smallerThanID] * (1 - (shells_radii[r2_smallerThanID, 2] ^ 3 - r2 ^ 3) / (shells_radii[r2_smallerThanID, 2] ^ 3 - shells_radii[r2_smallerThanID, 1] ^ 3))
        end
        
        if r1_smallerThanID == r2_smallerThanID
            expandedShells_mass[i] -= shells_mass[r1_smallerThanID]
        elseif r2_smallerThanID - r1_smallerThanID > 1
            expandedShells_mass[i] += sum(shells_mass[r1_smallerThanID + 1:r2_smallerThanID - 1])
        end
    end

    return expandedShells_mass
end


# For printing mass arrays
function printToFile(shells_radii, shells_mass, fileName, G)
    f = open(fileName, "w")
    
    shells_rho = zeros(size(shells_radii, 1))  # Shell density
    shells_enclosedMass = zeros(size(shells_radii, 1))  # Enclosed mass
    shells_avgRho = zeros(size(shells_radii, 1))  # Average density
    shells_Vcir = zeros(size(shells_radii, 1))  # Circular velocity
    for i in 1:size(shells_rho, 1)
        shells_rho[i] = shells_mass[i] / (shells_radii[i, 2] ^ 3 - shells_radii[i, 1] ^ 3) / (4 / 3 * pi)
        shells_enclosedMass[i] = sum(shells_mass[1:i])
        shells_avgRho[i] = shells_enclosedMass[i] / shells_radii[i, 2] ^ 3 / (4 / 3 * pi)
        shells_Vcir[i] = sqrt(G * shells_enclosedMass[i] / shells_radii[i, 2]) 
    end

    for i in 1:size(shells_radii, 1)
        println(f, shells_radii[i, 1], "\t", shells_radii[i, 2], "\t", shells_radii[i, 3], "\t", shells_mass[i], "\t", shells_rho[i], "\t", shells_enclosedMass[i], "\t", shells_avgRho[i], "\t", shells_Vcir[i])
    end
    
    close(f)
    return nothing
end


# For printing GPE arrays
function printToFile_GPE(Tshells_radii, Tshells_GPE, fileName)
    f = open(fileName, "w")

    for i in 1:size(Tshells_radii, 1)
        println(f, Tshells_radii[i, 1], "\t", Tshells_radii[i, 2], "\t", Tshells_radii[i, 3], "\t", Tshells_GPE[i])
    end

    close(f)
    return nothing
end

import math
import stormpy


# source: https://github.com/prismmodelchecker/prism/blob/master/prism/src/explicit/FoxGlynn.java
# https://github.com/prismmodelchecker/prism/blob/master/prism/src/explicit/CTMCModelChecker.java

def finder(q_tmax, accuracy, m, overflow, factor):
    sqrtpi = math.sqrt(math.pi)
    sqrt2 = math.sqrt(2)
    sqrtq = math.sqrt(q_tmax)
    aq = (1.0 + 1.0 / q_tmax) * math.exp(0.0625) * sqrt2
    bq = (1.0 + 1.0 / q_tmax) * math.exp(0.125 / q_tmax)

    lower_k_1 = 1.0 / (2.0 * sqrt2 * q_tmax)
    upper_k_1 = sqrtq / (2.0 * sqrt2)

    k = lower_k_1
    while k <= upper_k_1:
        dkl = 1.0 / (1 - math.exp(-(2.0 / 9.0) * (k * sqrt2 * sqrtq + 1.5)))
        res = aq * dkl * math.exp(-k * k / 2.0) / (k * sqrt2 * sqrtpi)
        if res <= accuracy / 2.0:
            break
        k = k + 4 if k == lower_k_1 else k + 1

    if k > upper_k_1:
        k = upper_k_1

    right = int(math.ceil(m + k * sqrt2 * sqrtq + 1.5))

    lower_k_2 = 1.0 / (sqrt2 * sqrtq)
    k = lower_k_2
    while True:
        res = bq * math.exp(-k * k / 2.0) / (k * sqrt2 * sqrtpi)
        if res <= accuracy / 2.0:
            break

    left = int(m - k * sqrtq - 1.5)

    wm = overflow / (factor * (right - left))
    weights = [0.0 for _ in range(right - left + 1)]
    weights[m - left] = wm

    return weights, left, right


def fox_glynn_algorithm(q_tmax, underflow, overflow, accuracy):
    if q_tmax == 0.0:
        raise ValueError("Overflow: q_tmax = time * max exit rate = 0.0")
    if accuracy < 1e-10:
        raise ValueError("Overflow: accuracy smaller than Fox-Glynn allows (must be > 1e-10)")
    if q_tmax < 400:
        expcoef = math.exp(-q_tmax)
        lastval = 1
        accum = lastval
        desval = (1 - (accuracy / 2.0)) / expcoef
        w = [lastval * expcoef]

        k = 1
        while True:
            lastval = lastval * q_tmax / k
            accum = accum + lastval
            w.append(lastval * expcoef)
            k = k + 1
            if accum >= desval:
                break
        left = 0
        right = k - 1
        weights = w
        total_weight = 1.0
    else:
        factor = 1e+10
        m = int(q_tmax)
        weights, left, right = finder(q_tmax, accuracy, m, overflow, factor)
        for j in range(m, left, -1):
            weights[j - 1 - left] = (j / q_tmax) * weights[j - left]
        for j in range(m, right, 1):
            weights[j + 1 - left] = (q_tmax / (j + 1)) * weights[j - left]
        total_weight = 0.0
        s = left
        t = right
        while s < t:
            if weights[s - left] <= weights[t - left]:
                total_weight = total_weight + weights[s - left]
                s = s + 1
            else:
                total_weight = total_weight + weights[t - left]
                t = t - 1
        total_weight = total_weight + weights[s - left]

    return weights, left, right, total_weight


if __name__ == '__main__':
    # build ctmc
    builder = stormpy.SparseMatrixBuilder(rows=4, columns=4, entries=6, force_dimensions=False,
                                          has_custom_row_grouping=False)
    builder.add_next_value(row=0, column=1, value=1.5)
    builder.add_next_value(row=1, column=0, value=3)
    builder.add_next_value(row=1, column=2, value=1.5)
    builder.add_next_value(row=2, column=1, value=3)
    builder.add_next_value(row=2, column=3, value=1.5)
    builder.add_next_value(row=3, column=2, value=3)
    transition_matrix = builder.build()

    state_labeling = stormpy.storage.StateLabeling(4)
    labels = {'empty', 'full'}
    for label in labels:
        state_labeling.add_label(label)

    state_labeling.add_label_to_state('empty', 0)
    state_labeling.add_label_to_state('full', 3)

    exit_rates = [1.5, 4.5, 4.5, 3.0]

    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                               rate_transitions=True)
    components.exit_rates = exit_rates
    ctmc = stormpy.storage.SparseCtmc(components)
    formula = "P=? [ true U[0, 7.5] \"full\" ]"
    properties = stormpy.parse_properties(formula)
    result = stormpy.model_checking(ctmc, properties[0])
    assert result.result_for_all_states
    tmp_y = [[result.at(i)] for i in range(ctmc.nr_states)]
    print(tmp_y)

    # do fox glynn
    qt = 7.5 * 4.5
    weights, left, right, total_weight = fox_glynn_algorithm(qt, 1.0e-300, 1.0e+300, 1.0e-10)
    for i in range(left, right + 1):
        weights[i - left] = weights[i - left] / total_weight
    print(weights, left, right, total_weight)


    #initialize
    target = [3] # state 3 satisfies phi 2 (full)
    nonAbs = [0, 1, 2] # only state 3 should be absorbing
    Abs = [3]
    soln = [0, 0, 0, 1]
    soln2 = [0, 0, 0, 1]
    sums = [0, 0, 0, 0]
    P = [[2/3, 1/3, 0, 0], [2/3, 0, 1/3, 0], [0, 2/3, 0, 1/3], [0, 0, 2/3, 1/3]]

    #0th element of summation


    '''
    iters = 1
    while(iters <= right):
        for s in nonAbs:
            soln2[s] = sum([P[s][j] * soln[j] for j in range(ctmc.nr_states)])
        tmpsoln = soln
        soln = soln2
        soln2 = tmpsoln
        if iters >= left:
            for i in range(ctmc.nr_states):
                sums[i] = sums[i] + weights[iters-left] * soln[i]
        iters = iters + 1
    '''
    if left == 0:
        for i in range(ctmc.nr_states):
            sums[i] = sums[i] + weights[0] * soln2[i]

    iters = 1
    while iters <= right:
        print(soln)
        for s in nonAbs:
            soln2[s] = sum([P[s][j] * soln[j] for j in range(ctmc.nr_states)])
        if iters >= left:
            for i in range(ctmc.nr_states):
                sums[i] = sums[i] + weights[iters - left] * soln2[i]
        iters = iters + 1
        soln = soln2.copy()

    final_sol = sums
    print(final_sol)


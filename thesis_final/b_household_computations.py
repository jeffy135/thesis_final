import numpy as np
import pandas as pd
import numba
import scipy
from numba import njit, prange
from interpolation import interp
from quantecon.optimize import nelder_mead

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# numba.config.NUMBA_DEFAULT_NUM_THREADS = 15       # @@Parallel
import time

# Define fixed parameters
discount = 0.85
city_count = 5 #@@citycount
theta = 0.6
delta = 0.01
## read in distance dataframe
log_distance_df = pd.read_csv("_data/output/log_distances.csv", index_col=0).iloc[:city_count, :city_count]
log_distance_array = log_distance_df.values


################################################################
#################             Compution tools                #####################
################################################################

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func: %r   took %2.4f  sec' % (f.__name__, te - ts))
        return result

    return timed

# @njit
def tool_numba_isnan(x): return np.isnan(x)


# @njit
def tool_approxZero(array):
    _ary = np.where(array < 0.01, 1e-15, array)
    return (_ary)


# @njit
def tool_migration_cost(migration_prop, distance = log_distance_array): return distance*migration_prop


# @njit
def tool_observed_prices(city_count, price_vector, iceberg_porp, distance = log_distance_array):
    nd_prices_origin = price_vector[:city_count]
    nd_prices_origin_mtx = np.multiply(np.ones(city_count).reshape(city_count, 1), nd_prices_origin)
    # iceberg = distance * iceberg_porp
    iceberg_costs = np.ones(city_count ** 2).reshape(city_count, city_count) + distance*iceberg_porp

    observed_costs = np.multiply(iceberg_costs, nd_prices_origin)
    return (observed_costs)

################################################################
#################             Demand Compute               #####################
################################################################

# @njit
def household_utility_givenLoc(consumption_bundle, individual_attributes, location_t1):
    CONSperiod_t0 = consumption_bundle[0:city_count]
    CONSperiod_t1 = consumption_bundle[city_count:2 * city_count]
    HOUSperiod_t1 = consumption_bundle[2 * city_count:3 * city_count]
    Q = consumption_bundle[3 * city_count:3 * city_count + 1]
    bonds = consumption_bundle[3 * city_count + 1: 3 * city_count + 2]
    tau_t1 = consumption_bundle[3 * city_count + 2:]

    HOUSperiod_t0 = individual_attributes[0:city_count]
    tastes = individual_attributes[city_count:2 * city_count + 1]
    location_t0 = individual_attributes[2 * city_count + 1:3 * city_count + 1]
    HP_prior = individual_attributes[3 * city_count + 1:]

    # First I must make sure the zeros in each consumption is non-zero.
    CONSperiod_t0 = tool_approxZero(CONSperiod_t0)
    CONSperiod_t1 = tool_approxZero(CONSperiod_t1)
    HOUSperiod_t1 = tool_approxZero(HOUSperiod_t1)

    location_t1 = tool_approxZero(location_t1)

    period_0_util = np.dot(tastes[:city_count], np.log(CONSperiod_t0)) + tastes[city_count] * np.log(tool_approxZero(np.dot(location_t0, HOUSperiod_t0)))
    period_1_util = np.dot(tastes[:city_count], np.log(CONSperiod_t1)) + tastes[city_count] * np.log(tool_approxZero(np.dot(location_t1, HOUSperiod_t1)))

    return period_0_util + discount * period_1_util


# @njit
def household_calculate_feasible_givenLoc(price_vector, individual_attributes, investment_grid, location_t1, migration_prop, iceberg_prop):
    # Unpack global prices
    _nd_price_origins = price_vector[0:city_count]
    _hpt0 = price_vector[city_count:2 * city_count]
    _wages = price_vector[2 * city_count:3 * city_count]
    _interest = price_vector[3 * city_count]

    # Unpack individual attributes
    HOUSperiod_t0 = individual_attributes[0:city_count]
    tastes = individual_attributes[city_count:2 * city_count + 1]
    location_t0 = individual_attributes[2 * city_count + 1:3 * city_count + 1]
    HP_prior = individual_attributes[3 * city_count + 1:]

    tastes_ndgoods = tastes[:-1]
    tastes_housing = tastes[-1]

    # Define expected prices
    wages_observed_t0 = np.dot(_wages, location_t0)
    wages_observed_t1 = np.dot(_wages, location_t1)
    obs_nd_price_origins = tool_observed_prices(city_count, _nd_price_origins, iceberg_prop)
    nd_price_observed_t0 = np.dot(obs_nd_price_origins, location_t0)
    nd_price_observed_t1 = np.dot(obs_nd_price_origins, location_t1)
    hp_t1 = (HP_prior + 1) * _hpt0

    migrationCosts = location_t0 @ tool_migration_cost(migration_prop) @ location_t1

    # Unpack investment_grid. This includes the tau_vector (of length equal to number of cities) and the Q value (scalar)
    tau_t1 = investment_grid[:-1]
    Q = investment_grid[-1]

    # Computation
    ## Compute housing stock in next period
    HOUSperiod_t1 = HOUSperiod_t0 * (1 - delta) + tau_t1

    ref_index = np.nonzero(tastes_ndgoods)[0][0]  # first index of non-zero in tastes

    relativeConsumption_t0 = np.divide(tastes_ndgoods, nd_price_observed_t0)
    relativeConsumption_t1 = np.divide(tastes_ndgoods, nd_price_observed_t1)

    consumption_ref_t0 = (wages_observed_t0 + np.dot(_hpt0, HOUSperiod_t0) - np.dot(_hpt0, tau_t1) - \
                          Q + (1 - theta) * np.dot(_hpt0, HOUSperiod_t1) - migrationCosts) * \
                         (tastes_ndgoods[ref_index] / nd_price_observed_t0[ref_index]) * np.dot(nd_price_observed_t0, relativeConsumption_t0)
    consumption_t0 = consumption_ref_t0 * (nd_price_observed_t0[ref_index] / tastes_ndgoods[ref_index]) * relativeConsumption_t0

    consumption_ref_t1 = (wages_observed_t1 + (1 + _interest) * (Q - (1 - theta) * np.dot(_hpt0, HOUSperiod_t1)) + \
                          np.dot(hp_t1, HOUSperiod_t1)) * \
                         (tastes_ndgoods[ref_index] / nd_price_observed_t1[ref_index]) * np.dot(nd_price_observed_t1, relativeConsumption_t1)
    consumption_t1 = consumption_ref_t1 * (nd_price_observed_t1[ref_index] / tastes_ndgoods[ref_index]) * relativeConsumption_t1

    bonds = Q - (1 - theta) * np.dot(_hpt0, HOUSperiod_t1)
    consumption_t0_lst = list(consumption_t0)
    consumption_t1_lst = list(consumption_t1)
    HOUSperiod_t1_lst = list(HOUSperiod_t1)
    Q_lst = [Q, bonds]
    tau_t1_lst = list(tau_t1)

    feasible_consumption_array = np.array(consumption_t0_lst + consumption_t1_lst + HOUSperiod_t1_lst + Q_lst + tau_t1_lst)

    return feasible_consumption_array

def find_opt_Q_giventaus_loc(tau_vector, Q_value, price_vector, individual_attributes, location_t1, migration_prop, iceberg_prop):
    currentPoint = np.array(list(tau_vector) + [Q_value])
    feasible_consumption_currentPoint = household_calculate_feasible_givenLoc(price_vector, individual_attributes, currentPoint, location_t1, migration_prop, iceberg_prop)
    utility_currentPoint = household_utility_givenLoc(feasible_consumption_currentPoint, individual_attributes, location_t1)
    return(utility_currentPoint)

# @njit
def household_generate_interps_givenLoc(price_vector, individual_attributes, gridded_city_index, location_t1, migration_prop, iceberg_prop):
    # Set upper limits to housing adjustments in each city.
    housingInv_upperbound = [100.] * city_count

    # Unpack global prices
    _hpt0 = price_vector[city_count:2 * city_count]
    _wages = price_vector[2 * city_count:3 * city_count]

    # Unpack individual attributes
    HOUSperiod_t0 = individual_attributes[0:city_count]
    location_t0 = individual_attributes[2 * city_count + 1:3 * city_count + 1]

    # Define expected prices
    wages_observed_t0 = np.dot(_wages, location_t0)

    tau_lower = []
    for city_index in gridded_city_index:
        lowerbound = -(HOUSperiod_t0[city_index] * (1 - delta))
        tau_lower.append(lowerbound)

    tau_upper = []
    for city_index in gridded_city_index:
        upperbound = housingInv_upperbound[city_index]
        tau_upper.append(upperbound)

    # Generate grid
    grid_tau_a = np.linspace(tau_lower[0], tau_upper[0], 4)  ## todo !! Change steps
    grid_tau_b = np.linspace(tau_lower[1], tau_upper[1], 4)
    grid_tau_c = np.linspace(tau_lower[2], tau_upper[2], 4)
    grid_tau_d = np.linspace(tau_lower[3], tau_upper[3], 4)
    grid_tau_e = np.linspace(tau_lower[4], tau_upper[4], 4)

    # Generate data holder for Q optimal and Utility optimal:
    util_opt_lst = []
    Q_opt_lst = []

    for tau_a in grid_tau_a:
        util_opt_lst_a = []
        Q_opt_lst_a = []
        for tau_b in grid_tau_b:
            util_opt_lst_ab = []
            Q_opt_lst_ab = []
            for tau_c in grid_tau_c:
                util_opt_lst_abc = []
                Q_opt_lst_abc = []
                for tau_d in grid_tau_d:
                    util_opt_lst_abcd = []
                    Q_opt_lst_abcd = []

                    for tau_e in grid_tau_e:
                        tau_vector = np.zeros(city_count)
                        adj_tau = [0., 0., 0., 0., 0.]
                        # re-create the tau vector used for adjusting housing stock
                        adj_tau[0] = tau_a
                        adj_tau[1] = tau_b
                        adj_tau[2] = tau_c
                        adj_tau[3] = tau_d
                        adj_tau[4] = tau_e

                        place_hold_counter = 0
                        for city_index in gridded_city_index:
                            tau_vector[city_index] = adj_tau[place_hold_counter]
                            place_hold_counter += 1

                        # Define the new housing stock given the choices for stock adjustments
                        HOUSperiod_t1 = HOUSperiod_t0 * (1 - delta) + tau_vector

                        # Define the upperbound for Q
                        Q_upperbound = wages_observed_t0 + (1 - theta) * np.dot(_hpt0, HOUSperiod_t1)

                        # Create Q grid for given tau_vector
                        linspace_grid_Q = np.linspace(0, Q_upperbound ** (1 / 2), 10)
                        grid_Q = linspace_grid_Q ** 2

                        # # define empty list of Q optimals and the corresponding utilities
                        # Q_opt_pt = -3.
                        # util_opt_pt = 0.
                        # for Q in grid_Q:
                        #     currentPoint = np.array(list(tau_vector) + [Q])
                        #     feasible_consumption_currentPoint = household_calculate_feasible_givenLoc(price_vector, individual_attributes, currentPoint, location_t1, migration_prop, iceberg_prop)
                        #     utility_currentPoint = household_utility_givenLoc(feasible_consumption_currentPoint, individual_attributes, location_t1)
                        #     if utility_currentPoint >= util_opt_pt:
                        #         util_opt_pt = utility_currentPoint
                        #         Q_opt_pt = Q
                        from joblib import Parallel, delayed
                        utility_generated_via_taus = Parallel(n_jobs=1)(delayed(find_opt_Q_giventaus_loc)(tau_vector, grid_Q[i], price_vector, individual_attributes, location_t1, migration_prop, iceberg_prop)for i in range(len(grid_Q)))
                        util_opt_pt = max(utility_generated_via_taus)
                        util_opt_indx = utility_generated_via_taus.index(util_opt_pt)
                        Q_opt_pt = grid_Q[util_opt_indx]

                        util_opt_lst_abcd.append(util_opt_pt)
                        Q_opt_lst_abcd.append(Q_opt_pt)

                    util_opt_lst_abc.append(util_opt_lst_abcd)
                    Q_opt_lst_abc.append(Q_opt_lst_abcd)

                util_opt_lst_ab.append(util_opt_lst_abc)
                Q_opt_lst_ab.append(Q_opt_lst_abc)

            util_opt_lst_a.append(util_opt_lst_ab)
            Q_opt_lst_a.append(Q_opt_lst_ab)

        util_opt_lst.append(util_opt_lst_a)
        Q_opt_lst.append(Q_opt_lst_a)

    util_opt_array = np.array(util_opt_lst)
    Q_opt_array = np.array(Q_opt_lst)

    return (util_opt_array, Q_opt_array, (grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e), (tau_lower, tau_upper))


# @njit
def utility_objective_givenLoc(x, *args):
    grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e, util_data_givenLoc = args

    u_func = interp(grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e, util_data_givenLoc, x)

    return u_func


# @njit
def household_opt_givenLoc(price_vector, individual_attributes, gridded_city_index, location_t1, migration_prop, iceberg_prop):
    util_data_givenLoc, Q_data_givenLoc, tau_pts_givenLoc, tau_bounds = household_generate_interps_givenLoc(price_vector, individual_attributes, gridded_city_index, location_t1, migration_prop, iceberg_prop)

    # Now I will do the interpolation that has bothered me since
    grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e = tau_pts_givenLoc

    q_func = lambda x: interp(grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e, Q_data_givenLoc, x)

    x0 =np.zeros(5) ## Set to 5 for number of choice cities

    # Set bounds for investments
    tau_lower_bound, tau_upper_bound = tau_bounds
    tau_bounds_array = np.array([tau_lower_bound, tau_upper_bound]).T

    tauQgrid_args = (grid_tau_a, grid_tau_b, grid_tau_c, grid_tau_d, grid_tau_e, util_data_givenLoc)

    util_optimizer = scipy.optimize.minimize(utility_objective_givenLoc, x0, args=tauQgrid_args)
    # util_optimizer = nelder_mead(utility_objective_givenLoc, x0, bounds=tau_bounds_array, args=tauQgrid_args)

    OPTIMAL_utility_givenLoc, OPTIMAL_tau_adj_t1_givenLoc = util_optimizer.fun, util_optimizer.x

    # calculate the optimal q calculated via interpolation
    OPTIMAL_q = q_func(OPTIMAL_tau_adj_t1_givenLoc)

    # change optimal tau_t1 from 5 into proper length, filling the non-gridded cities tau as 0
    OPTIMAL_tau_citylen_t1_givenLoc = np.zeros(city_count)

    place_hold_counter = 0
    for city_index in gridded_city_index:
        OPTIMAL_tau_citylen_t1_givenLoc[city_index] = OPTIMAL_tau_adj_t1_givenLoc[place_hold_counter]
        place_hold_counter += 1

    # calculate the optimal consumption
    OPTIMAL_investment_grid = np.array(list(OPTIMAL_tau_citylen_t1_givenLoc) + [OPTIMAL_q])
    OPTIMAL_consumptions_givenLoc = household_calculate_feasible_givenLoc(price_vector, individual_attributes, OPTIMAL_investment_grid, location_t1, migration_prop, iceberg_prop)

    return OPTIMAL_utility_givenLoc, OPTIMAL_consumptions_givenLoc


# @njit
def household_OPT(price_vector, individual_attributes, gridded_city_index, migration_prop, iceberg_prop):
    OPTIMAL_consumptions = np.empty(city_count * 4 + 1)
    OPTIMAL_utility_level = 0.
    OPTIMAL_location_t1 = [0.] * city_count

    for city_index in range(city_count):
        city_indicator = [0.] * city_count
        city_indicator[city_index] = 1.
        city_indicator_array = np.array(city_indicator)

        OPTIMAL_utility_givenLoc, OPTIMAL_consumptions_givenLoc = household_opt_givenLoc(price_vector, individual_attributes, gridded_city_index, city_indicator_array, migration_prop, iceberg_prop)

        if OPTIMAL_utility_givenLoc >= OPTIMAL_utility_level:
            OPTIMAL_utility_level = OPTIMAL_utility_givenLoc
            OPTIMAL_consumptions = OPTIMAL_consumptions_givenLoc
            OPTIMAL_location_t1 = city_indicator

    OPTIMAL_choice = (OPTIMAL_consumptions, OPTIMAL_location_t1)
    return (OPTIMAL_utility_level, OPTIMAL_choice)


# @njit #(parallel=True) #@@Parallel
def add_up_demands(world_data):
    # total_consumption = np.zeros(city_count)
    # total_housingtau = np.zeros(city_count)
    # total_bonds = 0.
    def individual_parse_info(individual_data, dname):
        util, (consumption, location) = individual_data

        if dname == "consumption":
            indiv_consumt0 = consumption[:city_count]
            return indiv_consumt0
        elif dname == "housing":
            indiv_houstau = consumption[-city_count:]
            return indiv_houstau
        elif dname =="bonds":
            indiv_bonds = consumption[3 * city_count + 1]
            return indiv_bonds

    from joblib import Parallel, delayed
    consumption_lst = Parallel(n_jobs=2, backend="multiprocessing")(delayed(individual_parse_info)(world_data[i], 'consumption') for i in range(len(world_data)))
    housing_lst = Parallel(n_jobs=2, backend="multiprocessing")(delayed(individual_parse_info)(world_data[i], 'housing') for i in range(len(world_data)))
    bonds_lst = Parallel(n_jobs=2, backend="multiprocessing")(delayed(individual_parse_info)(world_data[i], 'bonds') for i in range(len(world_data)))

    total_consumption = sum(consumption_lst)
    total_housingtau = sum(housing_lst)
    total_bonds = sum(bonds_lst)

    # for individual_index in range(len(world_data)):  #prange #@@Parallel
    #     individual_data = world_data[individual_index]
    #
    #     util, (consumption, location) = individual_data
    #
    #     # i only care about total demand in this period.
    #     indiv_consumt0 = consumption[:city_count]
    #
    #     # i only care about housing purchases this period.
    #     indiv_houstau = consumption[-city_count:]
    #
    #     # i only care about the savings
    #     indiv_bonds = consumption[3 * city_count + 1]
    #
    #     total_consumption += indiv_consumt0
    #     total_housingtau += indiv_houstau
    #     total_bonds += indiv_bonds
    return total_consumption, total_housingtau, total_bonds


# @njit # (parallel=True) #@@Parallel
def find_total_demand(world_attributes, prices, migration_prop, iceberg_prop):
    from joblib import Parallel, delayed
    all_individual_demands_p = Parallel(n_jobs=2, backend="multiprocessing")(delayed(household_OPT)(prices, world_attributes[i][0], world_attributes[i][1], migration_prop, iceberg_prop) for i in range(len(world_attributes)))

    # all_individual_demands_4 = []
    # for i in range(len(world_attributes)): #prange #@@Parallel
    #     indiv_input, grid_5cities = world_attributes[i]
    #     out = household_OPT(prices, indiv_input, grid_5cities, migration_prop, iceberg_prop)
    #     all_individual_demands_4.append(out)

    return add_up_demands(all_individual_demands_p)


# @njit #(parallel=True)#@@Parallel
def find_all_individual_demand(world_attributes, prices, migration_prop, iceberg_prop):
    from joblib import Parallel, delayed
    all_individual_demands_p = Parallel(n_jobs=2, backend="multiprocessing")(delayed(household_OPT)(prices, world_attributes[i][0], world_attributes[i][1], migration_prop, iceberg_prop) for i in range(len(world_attributes)))

    # all_individual_demands_4 = []
    # for i in range(len(world_attributes)): #prange#@@Parallel
    #     indiv_input, grid_5cities = world_attributes[i]
    #     out = household_OPT(prices, indiv_input, grid_5cities, migration_prop, iceberg_prop)
    #     all_individual_demands_4.append(out)

    return all_individual_demands_p
################################################################
#################              Supply Compute                 #####################
################################################################
# @njit
def find_total_supply(prices, world_attributes, total_demand, raw_productivity, supply_parameters):
    # Unpack supply parameters
    tfp_loc = supply_parameters[0]
    agglo_param = supply_parameters[1]
    a = supply_parameters[2]
    b = supply_parameters[3]
    residential_land_price = supply_parameters[4:city_count+4]
    production_land_price = supply_parameters[city_count+4:]

    # Unpack global prices
    _nd_price_origins = prices[0:city_count]
    _hpt0 = prices[city_count:2 * city_count]
    _wages = prices[2 * city_count:3 * city_count]
    _interest = prices[3 * city_count:]

    # Find current world_population
    current_world_population = np.zeros(city_count)
    for individual_attributes in world_attributes:
        current_world_population += individual_attributes[0][2*city_count+1:3*city_count+1]  # @ Change range

    # Find household ratio
    ndgood_demand, houstau_demand, bond_demand = total_demand
    _housing_nd_ratio = houstau_demand / tool_approxZero(ndgood_demand)

    # Compute observed productivity
    city_observed_tfp = agglo_param * np.log(tool_approxZero(current_world_population)) + raw_productivity

    # Compute the profit_cost_scalar for both sectors
    r_hs = (_hpt0 * (1 - b) / residential_land_price) ** ((1 - b) / b)
    r_nd = (_nd_price_origins * (1 - a) * city_observed_tfp / production_land_price) ** ((1 - a) / a)

    # Compute labor share of nd production sector
    nd_labor_share = r_hs / (_housing_nd_ratio * r_nd + r_hs)

    # Compute labor and land for each sector
    labor_hs = current_world_population * (1 - nd_labor_share)
    land_hs = (_hpt0 * (1 - b) / residential_land_price) ** (1 / b) * labor_hs

    labor_nd = current_world_population * nd_labor_share
    land_nd = (_nd_price_origins * (1 - a) * city_observed_tfp / production_land_price) ** (1 / a) * labor_nd

    ndgood_supply = city_observed_tfp * labor_nd ** a * land_nd ** (1 - a)
    housing_supply = labor_hs ** b * land_hs ** (1 - b)

    # Compute wage FOC
    ## Using either sector should yield the same results...so why not just the nd industry
    nd_implied_wages = _nd_price_origins * a * city_observed_tfp * (land_nd / tool_approxZero(labor_nd)) ** (1 - a)

    return ndgood_supply, housing_supply, nd_implied_wages

################################################################
#################              Market Clearing                 #####################
################################################################
# @njit#(parallel=True) #@@Parallel
def market_clearing_objective(prices, *args):
    # Unpack arguments
    world_attributes, raw_productivity, supply_parameters, migration_prop, iceberg_prop = args

    # Unpack prices
    _nd_price_origins = prices[0:city_count]
    _hpt0 = prices[city_count:2 * city_count]
    _wages = prices[2 * city_count:3 * city_count]
    _interest = prices[3 * city_count:]

    # First find demand
    tot_demand = find_total_demand(world_attributes, prices, migration_prop, iceberg_prop)
    # @@ Remove
    # print("Total demand: ", tot_demand)
    ## unpack demand
    ndgood_demand, houstau_demand, bond_demand = tot_demand

    # Second find supply
    ndgood_supply, houstau_supply, nd_implied_wages = find_total_supply(prices, world_attributes, tot_demand, raw_productivity, supply_parameters)

    clear_ndgood = ndgood_supply - ndgood_demand
    clear_houstau = houstau_supply - houstau_demand
    clear_labor = nd_implied_wages - _wages

    clear_bonds_lst = [bond_demand]
    clear_ndgood_lst = list(clear_ndgood)
    clear_houstau_lst = list(clear_houstau)
    clear_labor_lst = list(clear_labor)

    clear_MARKET_lst = clear_bonds_lst + clear_ndgood_lst + clear_houstau_lst + clear_labor_lst

    clear_MARKET_arr = np.array(clear_MARKET_lst)
    clear_MARKET_arr0 = np.empty(clear_MARKET_arr.shape)

    for i in range(len(clear_MARKET_arr)): #prange #@@Parallel
        if tool_numba_isnan(clear_MARKET_arr[i]):
            clear_MARKET_arr0[i]=0.
        else:
            clear_MARKET_arr0[i]=clear_MARKET_arr[i]
    # print("Market Status: ", clear_MARKET_arr0)
    return clear_MARKET_arr0

# @njit#(parallel=True) #@@Parallel
def market_clearing_pseudo(prices, *args):
        # Unpack arguments
        world_attributes, raw_productivity, supply_parameters, migration_prop, iceberg_prop = args

        ## @@"CALL BACK"
        # print('------iter--------')
        # print("Current Price", prices)
        # Unpack prices
        _nd_price_origins = prices[0:city_count]
        _hpt0 = prices[city_count:2 * city_count]
        _wages = prices[2 * city_count:3 * city_count]
        _interest = prices[3 * city_count:]

        # First find demand
        tot_demand = find_total_demand(world_attributes, prices, migration_prop, iceberg_prop)
        # @@ Remove
        # print("Total demand: ", tot_demand)
        ## unpack demand
        ndgood_demand, houstau_demand, bond_demand = tot_demand

        # Second find supply
        ndgood_supply, houstau_supply, nd_implied_wages = find_total_supply(prices, world_attributes, tot_demand, raw_productivity, supply_parameters)

        clear_ndgood = ndgood_supply - ndgood_demand
        clear_houstau = houstau_supply - houstau_demand
        clear_labor = nd_implied_wages - _wages

        clear_bonds_lst = [bond_demand]
        clear_ndgood_lst = list(clear_ndgood)
        clear_houstau_lst = list(clear_houstau)
        clear_labor_lst = list(clear_labor)

        clear_MARKET_lst = clear_bonds_lst + clear_ndgood_lst + clear_houstau_lst + clear_labor_lst

        clear_MARKET_arr = np.array(clear_MARKET_lst)
        clear_MARKET_arr0 = np.empty(clear_MARKET_arr.shape)

        for i in range(len(clear_MARKET_arr)): #prange #@@Parallel
            if tool_numba_isnan(clear_MARKET_arr[i]):
                clear_MARKET_arr0[i] = 0.
            else:
                clear_MARKET_arr0[i] = clear_MARKET_arr[i]
        L2_norm =  - np.linalg.norm(clear_MARKET_arr0)
        # print("Price: ", prices)
        # print("Norm: ", L2_norm)
        return L2_norm


if __name__ == '__main__':
    start_time = time.time()

    ## @@ When I SMM these change
    migration_proportion = 8.  # @@SMM
    iceberg_proportion = 0.02  # @@SMM


    from a_initiate_population import  InitPopulation
    import scipy.stats as sts
    population = InitPopulation()
    world_atts = population.Generate_World()


    tfp_loc = 1 #@@SMM
    agglo_param = 0.042 #@@SMM
    alpha = 0.4105030451244033
    beta = 0.25647330109680655

    residential_land_price_df = pd.read_csv("_data\output\scoped_residential.csv")
    residential_land_price_t = np.array(residential_land_price_df['2010'][:city_count])
    residential_land_price_lst = list(residential_land_price_t)

    production_land_prices_df = pd.read_csv("_data\output\scoped_industry.csv")
    production_land_price_t = np.array(production_land_prices_df['2010'][:city_count])
    production_land_price_lst = list(production_land_price_t)
    supply_parameters = np.array([tfp_loc, agglo_param, alpha, beta]+residential_land_price_lst + production_land_price_lst) # @@?? might need to initiate this when I am SMM-ing

    ## Set up raw productivity
    np.random.seed(0)
    city_raw_productivity_percentile = np.random.random(city_count)

    city_raw_productivty_draws = sts.invweibull.ppf(city_raw_productivity_percentile, c=10, loc=tfp_loc, scale=1)

    ## Compute observed productivity
    if not all(x > 1.0 for x in city_raw_productivty_draws):
        city_raw_productivty_draws = city_raw_productivty_draws / city_raw_productivty_draws.min()

    # Create price vector
    hpt0_df = pd.read_csv("_data/output/scoped_home_prices.csv", index_col=0).loc[:, '2010'][:city_count]
    wages_df = pd.read_csv("_data/output/scoped_wages_avg.csv", index_col=0).loc[:, '2010'][:city_count]

    initial_nd_price_origins = np.ones(city_count) * 10000.
    initial_hpt0 = hpt0_df.values
    initial_wages = wages_df.values
    initial_interest = 0.5
    initial_PRICES = np.append(initial_nd_price_origins, np.append(initial_hpt0, np.append(initial_wages, initial_interest)))

    mkcl_args = (world_atts, city_raw_productivty_draws, supply_parameters, migration_proportion, iceberg_proportion)
    price_initiation_minimized = scipy.optimize.minimize(market_clearing_pseudo, initial_PRICES,method='Nelder-Mead', args=mkcl_args, options={'maxiter':2})
    # price_initiation_minimized = nelder_mead(market_clearing_pseudo, initial_PRICES, args=mkcl_args, max_iter=1) ##@@@iter

    np.savetxt("_computedResults/test_prices.csv", price_initiation_minimized.x, delimiter=",")
    print(price_initiation_minimized.x)

    onedude_choice = find_all_individual_demand(world_atts, price_initiation_minimized.x, migration_proportion, iceberg_proportion)
    print(onedude_choice)
    print(time.time()-start_time)
    # print(onedude_choice)
    # save_it_dict = {"Prices: ": price_initiation_minimized.x,
    #                 "Minimized Value (from zero): ": -price_initiation_minimized.fun,
    #                 "Success: ": bool(price_initiation_minimized.success),
    #                 "Number of Interations: ": price_initiation_minimized.nit}
    #
    # import csv
    # w = csv.writer(open("_computedResults/test_init.csv", "w"))
    # for key, val in save_it_dict.items():
    #     w.writerow([key, val])

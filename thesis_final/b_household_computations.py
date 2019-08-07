from joblib import Parallel, delayed
import scipy.optimize as optimize

from a_initiate_population import *
from b_household_computations import *

numba.config.NUMBA_DEFAULT_NUM_THREADS = 15         # @@Parallel

# World &  Economy constants
city_count = 5 #@@citycount
theta = 0.6
delta = 0.01
log_distance_df = pd.read_csv("_data/output/log_distances.csv", index_col=0).iloc[:city_count, :city_count]
log_distance_array = log_distance_df.values


def parse_world(old_world, old_prices, new_world, new_prices, update_belief, city_count, data_name):
    def _individual (old_individual, old_prices, new_individual, new_prices, update_belief, city_count, data_name):
        old_attributes, old_grid5_cities = old_individual
        # Old world looks like this: ((stock_t0, tastes, location, belief), grid_5_cities)
        #               the dimensions:             n          n+1       n              n              lst(5)

        new_util, (new_choice, new_location) = new_individual
        # New world looks like this: (new util, ((consumption_t0, consumption_t1, housingStock_t1, Q_val, bonds, housing_demand_tau), location_t1))
        #                the dimensions:            1                        n                              n                          n                     1           1                         n                            n

        # Isolate Q value
        new_Q = new_choice[3*city_count]

        # Isolate new_consumption
        new_consumptiont0 = new_choice[:city_count]

        # Isolate new_housing_demand_tau
        new_housing_demand_tau = new_choice[-city_count:]

        # Isolate new housing stock
        new_houseStock = new_choice[2*city_count:3*city_count]

        # Isolate tastes
        new_tastes = old_attributes[city_count:2 * city_count + 1] # Tastes don't change

        # Update beliefs
        #### The belief of annual house price fluctuation is between -0.9 and 2.0
        lower_belief = -0.9
        upper_belief = 2.

        old_beliefs = old_attributes[3*city_count+1:]

        if update_belief:
            old_implied_prob = (old_beliefs - lower_belief) / (upper_belief - lower_belief)
            old_houseprice = old_prices[city_count:2 * city_count]
            new_houseprice = new_prices[city_count:2 * city_count]
            delta_houseprice = new_houseprice / old_houseprice - 1
            new_implied_prob = 0.7 * old_implied_prob + 0.3 * (delta_houseprice - lower_belief) / (upper_belief - lower_belief)
            new_beliefs = new_implied_prob * upper_belief + (1-new_implied_prob)
        else:
            new_beliefs = old_beliefs

        if data_name =="utility":
            return new_util
        elif data_name == "population":
            return new_location
        elif data_name == "consumption":
            return new_consumptiont0
        elif data_name == "housing":
            return new_housing_demand_tau
        elif data_name == "housing stock":
            return new_houseStock
        elif data_name == "q":
            return new_Q
        elif data_name == "beliefs":
            return new_beliefs
        elif data_name == "individual atts":
            # Create "individual_atts"
            restructured_new_individual_atts = np.array(list(new_houseStock) + list(new_tastes) + list(new_location) + list(new_beliefs))
            new_grid_5cities = get_grid5cities(restructured_new_individual_atts, city_count)
            return (restructured_new_individual_atts, new_grid_5cities)
        else:
            raise ValueError("No such key")
    output_lst = []
    for individual_ind in range(len(old_world)):
        j = _individual(old_world[individual_ind], old_prices, new_world[individual_ind], new_prices, update_belief, city_count, data_name)
        output_lst.append(j)
    # output_lst = Parallel()(delayed(_individual)(old_world[i], old_prices, new_world[i], new_prices, update_belief,city_count, data_name)for i in range(len(old_world)))
    return output_lst

# @@ When I SMM these change

def smm_obj(x):
    migration_proportion = x[0]
    iceberg_proportion = x[1]
    tfp_loc = x[2]
    agglo_param = x[3]

    # Demand-side constants
    discount = 0.85


    # Supply-side constants
    alpha = 0.4105030451244033
    beta = 0.25647330109680655


    ## Set up tfp
    np.random.seed(0)
    city_raw_productivity_percentile = np.random.random(city_count)
    city_raw_productivty_draws = sts.invweibull.ppf(city_raw_productivity_percentile, c=10, loc=tfp_loc, scale=1)
    if not all(x > 1.0 for x in city_raw_productivty_draws): # removes values of tfp < 1 (this leads to negative natural log outputs)
        city_raw_productivty_draws = city_raw_productivty_draws / city_raw_productivty_draws.min()


    # initiate price array placeholders
    nxt_market_clearing_initial_prices = np.empty(3*city_count+1)

    period_world_data_anchor = []
    ################################################################
    ############              Computation for Periods              ####################
    ################################################################
    for period in range(6):
        if period==0:
            current_period_year = str(period + 2010)
            next_period_year = str(period+2011)
            print("initialize")
            # Initialize population
            Population = InitPopulation()
            period_world_data = Population.Generate_World()

            # Save demographic information
            world_housingStock = []
            world_tastes = []
            world_location = []
            world_beliefs = []
            for individual in period_world_data:
                world_housingStock.append(individual[0][0:city_count]) # indiv_housingStock
                world_tastes.append(individual[0][city_count:2 * city_count + 1]) # indiv_tastes
                world_location.append(individual[0][2 * city_count + 1:3 * city_count + 1]) # indiv_location
                world_beliefs.append(individual[0][3 * city_count + 1:]) # indiv_belief

            np.savetxt("_computedResults/period" + str(current_period_year) + "_housingStock.csv", world_housingStock, delimiter=",")
            np.savetxt("_computedResults/period" + str(current_period_year) + "_tastes.csv", world_tastes, delimiter=",")
            np.savetxt("_computedResults/period" + str(current_period_year) + "_population.csv", world_location, delimiter=",")
            np.savetxt("_computedResults/period" + str(current_period_year) + "_beliefs.csv", world_beliefs, delimiter=",")

            # Obtain period land prices
            residential_land_price_df = pd.read_csv("_data\output\scoped_residential.csv")
            residential_land_price = np.array(residential_land_price_df[current_period_year][:city_count])
            residential_land_price_lst = list(residential_land_price)

            production_land_prices_df = pd.read_csv("_data\output\scoped_industry.csv")
            production_land_price = np.array(production_land_prices_df[current_period_year][:city_count])
            production_land_price_lst = list(production_land_price)
            supply_parameters_t = np.array([tfp_loc, agglo_param, alpha, beta]+residential_land_price_lst + production_land_price_lst) # @@?? might need to initiate this when I am SMM-ing

            # Produce initial price vector.
            hpt0_df = pd.read_csv("_data/output/scoped_home_prices.csv", index_col=0).loc[:, current_period_year][:city_count]
            wages_df = pd.read_csv("_data/output/scoped_wages_avg.csv", index_col=0).loc[:, current_period_year][:city_count]

            initial_nd_price_origins = np.ones(city_count) * 10000.
            initial_hpt0 = hpt0_df.values
            initial_wages = wages_df.values
            initial_interest = 0.5
            initial_PRICES = np.append(initial_nd_price_origins, np.append(initial_hpt0, np.append(initial_wages, initial_interest)))

            # Execute market clearing
            mkcl_args = (period_world_data, city_raw_productivty_draws, supply_parameters_t, migration_proportion, iceberg_proportion)
            _market_clearing_info = nelder_mead(market_clearing_pseudo, initial_PRICES, args=mkcl_args, tol_f=0.01, tol_x=0.1, max_iter=1)  ##@@@iter
            print("market_clearing_objective complete")
            # _market_clearing_info = optimize.root(market_clearing_objective, initial_PRICES, args=mkcl_args, method='hybr', options={'maxfev':1, 'xtol':10.})

            # Reassign market prices
            ths_market_clearing_prices = _market_clearing_info.x
            period_market_clearing_ndprices = ths_market_clearing_prices[:city_count]
            period_market_clearing_hsprices = ths_market_clearing_prices[city_count:2*city_count]
            period_market_clearing_wages = ths_market_clearing_prices[2*city_count:3*city_count]
            period_market_clearing_interest = ths_market_clearing_prices[3 * city_count:3 * city_count+1]

            # Save price vector
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_ndpriceorigin.csv", period_market_clearing_ndprices, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_houseprices.csv", period_market_clearing_hsprices, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_wages.csv", period_market_clearing_wages, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_interestrate.csv", period_market_clearing_interest, delimiter=",")


            # @@? Must create a deep copy of the current world attributes to run here again.
            import copy
            period_world_data_recal = copy.deepcopy(period_world_data)
            # Generate all world data
            A_Whole_New_World = find_all_individual_demand(period_world_data_recal, ths_market_clearing_prices, migration_proportion, iceberg_proportion)

            # Parse the world data into managable chuncks
            clean_info_order = ['utility', 'population', 'consumption', 'housing', 'housing stock', 'q', 'beliefs', 'individual atts']
            update_belief = False
            # @@Parallel
            parsed_information = []
            for i in range(len(clean_info_order)):
                parsed_information.append(parse_world(period_world_data, nxt_market_clearing_initial_prices, A_Whole_New_World, ths_market_clearing_prices, update_belief, city_count,  clean_info_order[i]))

            # parsed_information = Parallel()(delayed(parse_world)(period_world_data, nxt_market_clearing_initial_prices, A_Whole_New_World, ths_market_clearing_prices, update_belief, city_count,  clean_info_order[i]) for i in range(len(clean_info_order)))

            ths_utility_data, ths_population_data, ths_consumption_data, ths_housingD_data, ths_housingS_data, ths_q_data, ths_beliefs_data, ths_individual_atts =  parsed_information

            # Save all data
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_utility.csv", ths_utility_data, delimiter=",") # utility is current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_population.csv", ths_population_data, delimiter=",") # population is migration, next period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_ndconsumption.csv", ths_consumption_data, delimiter=",") # consumption current period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_housingdemand.csv", ths_housingD_data, delimiter=",") # housing demand is current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_housingstock.csv", ths_housingS_data, delimiter=",") # housing stock is next period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_qdata.csv", ths_q_data, delimiter=",") # Q is for current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_utility.csv", ths_beliefs_data, delimiter=",") # beliefs is for next period

            nxt_market_clearing_initial_prices = ths_market_clearing_prices
            period_world_data_anchor = ths_individual_atts
        else:
            current_period_year = str(period + 2010)
            next_period_year = str(period+2011)
            print("continue" + str(current_period_year))

            # Initialize population, recalling from previous period
            period_world_data = period_world_data_anchor

            # Obtain period land prices
            residential_land_price_df = pd.read_csv("_data\output\scoped_residential.csv")
            residential_land_price = np.array(residential_land_price_df[current_period_year][:city_count])
            residential_land_price_lst = list(residential_land_price)

            production_land_prices_df = pd.read_csv("_data\output\scoped_industry.csv")
            production_land_price = np.array(production_land_prices_df[current_period_year][:city_count])
            production_land_price_lst = list(production_land_price)
            supply_parameters_t = np.array([tfp_loc, agglo_param, alpha, beta] + residential_land_price_lst + production_land_price_lst)  # @@?? might need to initiate this when I am SMM-ing

            # Produce initial price vector, recalling from previous period's market clearing price.
            initial_PRICES = nxt_market_clearing_initial_prices
            print("Begin market clearing")
            # Execute market clearing
            mkcl_args = (period_world_data, city_raw_productivty_draws, supply_parameters_t, migration_proportion, iceberg_proportion)
            _market_clearing_info = nelder_mead(market_clearing_pseudo, initial_PRICES, args=mkcl_args, max_iter=1)  ##@@@iter
            print("market_clearing_objective complete")

            # Reassign market prices
            ths_market_clearing_prices = _market_clearing_info.x
            period_market_clearing_ndprices = ths_market_clearing_prices[:city_count]
            period_market_clearing_hsprices = ths_market_clearing_prices[city_count:2 * city_count]
            period_market_clearing_wages = ths_market_clearing_prices[2 * city_count:3 * city_count]
            period_market_clearing_interest = ths_market_clearing_prices[3 * city_count:3 * city_count + 1]

            # Save price vector
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_ndpriceorigin.csv", period_market_clearing_ndprices, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_houseprices.csv", period_market_clearing_hsprices, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_wages.csv", period_market_clearing_wages, delimiter=",")
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_interestrate.csv", period_market_clearing_interest, delimiter=",")

            # @@? Must create a deep copy of the current world attributes to run here again.
            import copy

            period_world_data_recal = period_world_data #copy.deepcopy(period_world_data)
            # Generate all world data
            A_Whole_New_World = find_all_individual_demand(period_world_data_recal, ths_market_clearing_prices, migration_proportion, iceberg_proportion)

            # Parse the world data into managable chuncks
            clean_info_order = ['utility', 'population', 'consumption', 'housing', 'housing stock', 'q', 'beliefs', 'individual atts']
            update_belief = False
            # @@Parallel
            parsed_information = []
            for i in range(len(clean_info_order)):
                parsed_information.append(parse_world(period_world_data, nxt_market_clearing_initial_prices, A_Whole_New_World, ths_market_clearing_prices, update_belief, city_count,  clean_info_order[i]))
            # parsed_information = Parallel()(delayed(parse_world)(period_world_data, nxt_market_clearing_initial_prices, A_Whole_New_World, ths_market_clearing_prices, update_belief, city_count, clean_info_order[i]) for i in range(len(clean_info_order)))

            ths_utility_data, ths_population_data, ths_consumption_data, ths_housingD_data, ths_housingS_data, ths_q_data, ths_beliefs_data, ths_individual_atts = parsed_information

            # Save all data
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_utility.csv", ths_utility_data, delimiter=",")  # utility is current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_population.csv", ths_population_data, delimiter=",")  # population is migration, next period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_ndconsumption.csv", ths_consumption_data, delimiter=",")  # consumption current period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_housingdemand.csv", ths_housingD_data, delimiter=",")  # housing demand is current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_housingstock.csv", ths_housingS_data, delimiter=",")  # housing stock is next period
            np.savetxt("_computedResults/Period" + str(current_period_year) + "_qdata.csv", ths_q_data, delimiter=",")  # Q is for current period
            np.savetxt("_computedResults/Period" + str(next_period_year) + "_utility.csv", ths_beliefs_data, delimiter=",")  # beliefs is for next period

            nxt_market_clearing_initial_prices = ths_market_clearing_prices
            period_world_data_anchor = ths_individual_atts
            print("... completed ", current_period_year)

    # Compute data moments
    ## compute generated moments for wage changes
    array_wages_2010 = pd.read_csv("_computedResults\Period2010_wages.csv",header=None, index_col=False).values.T[0]
    array_wages_2011 = pd.read_csv("_computedResults\Period2011_wages.csv",header=None, index_col=False).values.T[0]
    array_wages_2012 = pd.read_csv("_computedResults\Period2012_wages.csv",header=None, index_col=False).values.T[0]
    array_wages_2013 = pd.read_csv("_computedResults\Period2013_wages.csv",header=None, index_col=False).values.T[0]
    array_wages_2014 = pd.read_csv("_computedResults\Period2014_wages.csv",header=None, index_col=False).values.T[0]
    array_wages_2015 = pd.read_csv("_computedResults\Period2015_wages.csv", header=None, index_col=False).values.T[0]

    lst_wages = np.array([array_wages_2010, array_wages_2011, array_wages_2012, array_wages_2013, array_wages_2014, array_wages_2015])
    dataframe_wages_horizontal = pd.DataFrame(lst_wages)

    dataframe_wages_abs = dataframe_wages_horizontal.T
    dataframe_wages_abs.columns = [y for y in range(2010,2016)]

    dataframe_wages_pct = dataframe_wages_abs.pct_change(axis=1)
    observed_data_wages = dataframe_wages_pct.iloc[:, 1:].values

    ## Compute generated moments for houseprice changes
    array_houseprices_2010 = pd.read_csv("_computedResults\Period2010_houseprices.csv",header=None, index_col=False).values.T[0]
    array_houseprices_2011 = pd.read_csv("_computedResults\Period2011_houseprices.csv",header=None, index_col=False).values.T[0]
    array_houseprices_2012 = pd.read_csv("_computedResults\Period2012_houseprices.csv",header=None, index_col=False).values.T[0]
    array_houseprices_2013 = pd.read_csv("_computedResults\Period2013_houseprices.csv",header=None, index_col=False).values.T[0]
    array_houseprices_2014 = pd.read_csv("_computedResults\Period2014_houseprices.csv",header=None, index_col=False).values.T[0]
    array_houseprices_2015 = pd.read_csv("_computedResults\Period2015_houseprices.csv",header=None, index_col=False).values.T[0]
    lst_housepreices = np.array([array_houseprices_2010, array_houseprices_2011, array_houseprices_2012, array_houseprices_2013, array_houseprices_2014, array_houseprices_2015])
    dataframe_houseprices_abs = pd.DataFrame(lst_housepreices).T
    dataframe_houseprices_abs.columns = [y for y in range(2010,2016)]

    dataframe_houseprices_pct = dataframe_houseprices_abs.pct_change(axis=1)
    observed_data_houseprices = dataframe_houseprices_pct.iloc[:, 1:].values

    ## Compute generated moments for gdp changes
    array_tothousingdemand_2010 = pd.read_csv("_computedResults\Period2010_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)
    array_tothousingdemand_2011 = pd.read_csv("_computedResults\Period2011_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)
    array_tothousingdemand_2012 = pd.read_csv("_computedResults\Period2012_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)
    array_tothousingdemand_2013 = pd.read_csv("_computedResults\Period2013_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)
    array_tothousingdemand_2014 = pd.read_csv("_computedResults\Period2014_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)
    array_tothousingdemand_2015 = pd.read_csv("_computedResults\Period2015_housingdemand.csv",header=None, index_col=False).T.sum(axis=1)

    lst_totalhousingdemand = [array_tothousingdemand_2010, array_tothousingdemand_2011, array_tothousingdemand_2012, array_tothousingdemand_2013, array_tothousingdemand_2014, array_tothousingdemand_2015]
    dataframe_totalhousingdemand_abs = pd.DataFrame(lst_totalhousingdemand).T
    dataframe_totalhousingdemand_abs.columns = [y for y in range(2010,2016)]

    dataframe_totalhsexpenditure_absvalues = (dataframe_totalhousingdemand_abs * dataframe_houseprices_abs).values

    array_totndconsumption_2010 = pd.read_csv("_computedResults\Period2010_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    array_totndconsumption_2011 = pd.read_csv("_computedResults\Period2011_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    array_totndconsumption_2012 = pd.read_csv("_computedResults\Period2012_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    array_totndconsumption_2013 = pd.read_csv("_computedResults\Period2013_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    array_totndconsumption_2014 = pd.read_csv("_computedResults\Period2014_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    array_totndconsumption_2015 = pd.read_csv("_computedResults\Period2015_ndconsumption.csv",header=None, index_col=False).T.sum(axis=1)
    lst_totndconsumption = [array_totndconsumption_2010, array_totndconsumption_2011, array_totndconsumption_2012, array_totndconsumption_2013, array_totndconsumption_2014, array_totndconsumption_2015]
    dataframe_ndconsumption_abs = pd.DataFrame(lst_totndconsumption).T
    dataframe_ndconsumption_abs.columns = [y for y in range(2010,2016)]

    array_ndpriceorigin_2010 = pd.read_csv("_computedResults\Period2010_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    array_ndpriceorigin_2011 = pd.read_csv("_computedResults\Period2011_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    array_ndpriceorigin_2012 = pd.read_csv("_computedResults\Period2012_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    array_ndpriceorigin_2013 = pd.read_csv("_computedResults\Period2013_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    array_ndpriceorigin_2014 = pd.read_csv("_computedResults\Period2014_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    array_ndpriceorigin_2015 = pd.read_csv("_computedResults\Period2015_ndpriceorigin.csv",header=None, index_col=False).values.T[0]
    lst_ndpriceorigin = [array_ndpriceorigin_2010, array_ndpriceorigin_2011, array_ndpriceorigin_2012, array_ndpriceorigin_2013, array_ndpriceorigin_2014, array_ndpriceorigin_2015]
    dataframe_ndpriceorigin_abs = pd.DataFrame(lst_ndpriceorigin).T
    dataframe_ndpriceorigin_abs.columns = [y for y in range(2010,2016)]

    dataframe_totalndexpenditure_absvalues = (dataframe_ndconsumption_abs * dataframe_ndpriceorigin_abs).values

    dataframe_gdp_absvalues = dataframe_totalndexpenditure_absvalues + dataframe_totalhsexpenditure_absvalues
    dataframe_gdp_abs = pd.DataFrame(dataframe_gdp_absvalues, columns=[y for y in range(2010,2016)])
    dataframe_gdp_pct = dataframe_gdp_abs.pct_change(axis=1)
    observed_data_gdp = dataframe_gdp_pct.iloc[:, 1:].values

    ## Compute generated moments for population changes
    array_population_2010 = pd.read_csv("_computedResults\Period2010_location.csv",header=None, index_col=False).T.sum(axis=1)
    array_population_2011 = pd.read_csv("_computedResults\Period2011_population.csv",header=None, index_col=False).T.sum(axis=1)
    array_population_2012 = pd.read_csv("_computedResults\Period2012_population.csv",header=None, index_col=False).T.sum(axis=1)
    array_population_2013 = pd.read_csv("_computedResults\Period2013_population.csv",header=None, index_col=False).T.sum(axis=1)
    array_population_2014 = pd.read_csv("_computedResults\Period2014_population.csv",header=None, index_col=False).T.sum(axis=1)
    array_population_2015 = pd.read_csv("_computedResults\Period2015_population.csv",header=None, index_col=False).T.sum(axis=1)
    lst_population = [array_population_2010, array_population_2011, array_population_2012, array_population_2013, array_population_2014, array_population_2015]
    dataframe_totalpopulation_abs = pd.DataFrame(lst_population).T
    dataframe_totalpopulation_abs.columns = [y for y in range(2010,2016)]
    dataframe_totalpopulation_pct = dataframe_totalpopulation_abs.pct_change(axis=1)

    observed_data_totalpopulation = dataframe_totalpopulation_pct.iloc[:, 1:].values

    # Read in the realworld data
    realworld_data_gdp = pd.read_csv("_data\output\perchg_gdp.csv").iloc[:city_count, 2:].values
    realworld_data_wages = pd.read_csv("_data\output\perchg_wages_avg.csv").iloc[:city_count, 2:].values
    realworld_data_houseprices = pd.read_csv("_data\output\perchg_home_prices.csv").iloc[:city_count, 2:].values
    realworld_data_population = pd.read_csv("_data\output\perchg_population.csv").iloc[:city_count, 2:].values

    # Compute distance between generated data and realworld data
    DataDistance_gdp = np.linalg.norm(np.nan_to_num(observed_data_gdp - realworld_data_gdp))
    DataDistance_wages = np.linalg.norm(np.nan_to_num(observed_data_wages - realworld_data_wages))
    DataDistance_houseprices = np.linalg.norm(np.nan_to_num(observed_data_houseprices - realworld_data_houseprices))
    DataDistance_population = np.linalg.norm(np.nan_to_num(observed_data_totalpopulation - realworld_data_population))

    print("DataDistance_gdp: ", DataDistance_gdp)
    print("DataDistance_wages: ", DataDistance_wages)
    print("DataDistance_houseprices: ", DataDistance_houseprices)
    print("DataDistance_population: ", DataDistance_population)

    DataDistance_Total = DataDistance_gdp + DataDistance_wages + DataDistance_houseprices + DataDistance_population
    print("DataDistance_Total: ", DataDistance_Total)
    return(-DataDistance_Total)

if __name__=="__main__":
    initial_smm_parameters = np.array([8., 0.02, 1, 0.042])
    # smm_obj(initial_smm_parameters)
    # initial_smm_parameters = np.array([7., 0.02, 1, 0.042])
    # smm_obj(initial_smm_parameters)
    print(optimize.minimize(smm_obj, initial_smm_parameters, method='Nelder-Mead', options={'maxiter':2}))

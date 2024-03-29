import numpy as np
import scipy.stats as sts
import pandas as pd


class InitPopulation():
    def __init__(self,
                 cities_and_population=[1,1,1,1,1],
                 taste_params=(0., 0.15, 1.8),
                 calibration=False,
                 initial_seed=0):
        self.city_population = cities_and_population
        self.cities = len(cities_and_population)
        self.taste_params = taste_params

        self.initial_seed = initial_seed
        self.calibrated = calibration

    def __generate_tastes(self, seed):
        loc, scale, c = self.taste_params
        # $$ Set Seed $$
        np.random.seed(seed)
        city_index = list(np.arange(self.cities).tolist())
        individual_tastes = np.zeros(self.cities)
        proceed_boolean = True
        while proceed_boolean:
            rand_taste = sts.invweibull.rvs(c=c, loc=loc, scale=scale)
            city_ = city_index.pop(np.random.randint(0, len(city_index)))
            if len(city_index) < 1 or sum(individual_tastes) + rand_taste >= 0.8:
                proceed_boolean = False
                rand_taste = 0.8 - sum(individual_tastes)
            else:
                pass
            individual_tastes[city_] = rand_taste
        individual_tastes = np.append(individual_tastes, np.array([[0.2]]))
        return individual_tastes

    def __generate_HP_pior(self, seed):
        np.random.seed(seed)
        individual_HP_prior = sts.truncnorm.rvs(-0.9, 2.0, size=self.cities)
        return individual_HP_prior

    def __generate_HOUSperiod_t0(self, city_index, seed):
        if self.calibrated:
            # Generate list of numbers (home ownership count) for each individual
            np.random.seed(seed)
            indiv_ownership = int(sts.expon.rvs(scale=0.4, loc=1))
            cities_of_interest = [line.rstrip('\n') for line in open(r"_data\output\included_cities.txt", 'r')]

            # Obtain house prices
            houseprice = pd.read_csv(r"_data\Data\singles\loc_home_prices.csv")
            dtdes = houseprice['点击这里以选择指标，然后再点击"更新"或"下载编辑"按钮更新或新增及移除数列']

            city_lst = []
            for text in dtdes:
                city_lst.append(text[text.rindex(":") + 1:].replace("市", ""))

            houseprice['City'] = city_lst
            houseprice = pd.DataFrame(houseprice[['City', '2009']]).set_index("City")
            houseprice2009 = np.array(houseprice.loc[cities_of_interest]).ravel()

            # Create wealth draws @@ Calibrated exogenously
            np.random.seed(seed)
            set_wealth_percentiles = np.random.random()

            # indiv_wealth = sts.lognorm.ppf(set_wealth_percentiles, s=1.51920997e+00, scale=4.77488100e+04, loc=2000 * 6.77)
            indiv_wealth = sts.lognorm.rvs(s=1.51920997e+00, scale=4.77488100e+04, loc=2000 * 6.77)

            # Generate city indices with housing stock
            np.random.seed(seed)
            city_ownership = [city_index] + list(np.random.choice(self.cities, size=indiv_ownership - 1, replace=False))

            # Generate housing portfolio shares of wealth
            np.random.seed(seed)
            housing_portfolio_shares = np.random.random(indiv_ownership)
            housing_portfolio_shares = housing_portfolio_shares / housing_portfolio_shares.sum()

            # Calculate wealth on each house
            housing_wealth_ = housing_portfolio_shares * indiv_wealth

            housing_wealth = np.zeros(self.cities)
            housing_wealth[city_ownership] = housing_wealth_
            housing_wealth_units = housing_wealth / houseprice2009

            return housing_wealth_units

        else:
            # $$ Set Seed $$
            np.random.seed(seed)
            location_t0 = self.__generate_location_t0(city_index)
            individual_HOUSperiod_t0 = np.random.random(self.cities) * 123 * [int(bool(f > 0.86)) for f in np.random.random(self.cities)] + \
                                       np.random.random() * 123 * location_t0
            return individual_HOUSperiod_t0

    def __generate_location_t0(self, city_index):
        individual_location = [0] * self.cities
        individual_location[city_index] = 1
        return np.array(individual_location)

    def __populate_one(self, city_index, individual_seed):
        individual_housing_stock_t0 = self.__generate_HOUSperiod_t0(city_index, individual_seed)
        individual_tastes = self.__generate_tastes(individual_seed)
        individual_location_t0 = self.__generate_location_t0(city_index)
        individual_beliefs = self.__generate_HP_pior(individual_seed)

        grid_5cities = []
        top_cities_descending = sorted(range(len(individual_beliefs)), key=lambda i: individual_beliefs[i], reverse=True)
        # Take away current city from the lot
        grid_5cities.append(top_cities_descending.pop(top_cities_descending.index(city_index)))

        # Take away top 3 and bottom 1
        grid_5cities += top_cities_descending[:3]
        grid_5cities.append(top_cities_descending[-1])

        individual_atts = np.array(list(individual_housing_stock_t0) + list(individual_tastes) + list(individual_location_t0) + list(individual_beliefs))

        return (individual_atts, grid_5cities)

    def __populate_city(self, city_index):
        """
        Given a city index, this function generates the individual attributes
        Args:
            city_index:

        Returns:

        """
        city_population = self.city_population[city_index]

        city_dweller_attributes = []
        city_initial_seed = self.initial_seed + sum(self.city_population[:city_index])
        for person in range(city_population):
            # $$ Some random rule to generate random seed $$
            individual_seed = city_initial_seed + person
            individual_att = self.__populate_one(city_index, individual_seed)
            city_dweller_attributes.append(individual_att)

        return city_dweller_attributes

    def Generate_World(self):
        world_attributes = []
        for city_ind in range(self.cities):
            city_att = self.__populate_city(city_ind)
            world_attributes += city_att
        return world_attributes

def get_grid5cities(individual_attribute, city_count):
    beliefs = individual_attribute[-city_count:]
    city_index = list(individual_attribute[-2*city_count:-city_count]).index(1.)
    grid_5cities = []
    top_cities_descending = sorted(range(len(beliefs)), key=lambda i: beliefs[i], reverse=True)
    # Take away current city from the lot
    grid_5cities.append(top_cities_descending.pop(top_cities_descending.index(city_index)))

    # Take away top 3 and bottom 1
    grid_5cities += top_cities_descending[:3]
    grid_5cities.append(top_cities_descending[-1])
    return grid_5cities

if __name__ == "__main__":
    import time
    start_time= time.time()
    j=InitPopulation(calibration=True)
    j.Generate_World()
    print(time.time()-start_time)

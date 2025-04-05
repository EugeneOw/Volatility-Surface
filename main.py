import numpy as np
import pandas as pd
import yfinance as yf
from typing import Union
from loguru import logger
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime as dt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import griddata


@dataclass
class IdvOption:
    implied_vol: float
    strike_price: float
    days_to_expr: int


class RetrieveData:
    def __init__(self, __ticker: str):
        """
        Attributes:
            self.__ticker - The ticker symbol
            self.__today_date - Contains today date
            self.__ticker_details - Should contain the Ticker class
            self.__ticker_curr_price - Current ticker price
            self.__expr_dates - Contain all the options expiry dates available to this ticker <self.__ticker>
            self.__expiry_date - Contain single expiry date that is not null
            self.__days_to_expr - Contains the days difference from today and expiry date of the option
            self.__option_chain - All the option trades of that have that particular expiry date
            self.__call_option_df - Contains the call options as a type <pd.dataFrame>
            self.__put_option_df - Contains the put options as a type <pd.DataFrame>
            self.__clean_call_df - Contains only the 'strike' and 'impliedVolatility' for the call options dataFrame
            self.__clean_put_df - Contains only the 'strike' and 'impliedVolatility' for the put options dataFrame
            self.__appr_call_option - A list that contains all the approved options in a dataclass
            self.__appr_put_option - A list that contains all the approved options in a dataclass

            self.__DAYS_MIN - Minimum no. of days that are allowed to be on the threshold
            self.__DAYS_MAX - Maximum no. of days that are allowed to be on the threshold
            self.__PRICE_MIN_PERCENT - Minimum price percent that is allowed to be on the threshold
            self.__PRICE_MAX_PERCENT - Maximum price percent that is allowed to be on the threshold
        """

        self.__today_date: dt.date = dt.today().date()
        self.__ticker_details = None
        self.__ticker_curr_price: float = 0.0
        self.__expr_dates: tuple[str, ...] = ()
        self.__expiry_date: dt.date = ""
        self.__days_to_expr: int = 0
        self.__option_chain: dataclass() = None
        self.__call_option_df: pd.DataFrame() = None
        self.__put_option_df: pd.DataFrame() = None
        self.__clean_call_df: pd.DataFrame() = None
        self.__clean_put_df: pd.DataFrame() = None
        self.__appr_call_option: list[dataclass()] = []
        self.__appr_put_option: list[dataclass()] = []

        self.__DAYS_MIN: int = 0
        self.__DAYS_MAX: int = 31
        self.__PRICE_MIN_PERCENT: float = 0.95
        self.__PRICE_MAX_PERCENT: float = 1.05

        if not self.__is_valid_ticker(__ticker):
            return
        self.__ticker: str = __ticker

        if not self.__get_ticker_details():
            return

        if not self.__get_expr_dates():
            return

        for __index, __expiry_date in enumerate(self.__expr_dates):
            if not self.__is_valid_expr_date(__expiry_date):
                return
            self.__expiry_date: dt.date = dt.strptime(__expiry_date, "%Y-%m-%d").date()

            if not self.__is_within_margin_date():
                break

            if not self.__get_option_chain():
                return

            if not self.__get_call_option():
                return

            if not self.__get_put_option():
                return

            if not self.__filter_call_df():
                return

            if not self.__filter_put_df():
                return

            self.__filter_call_s_price()
            self.__filter_put_s_price()

    @staticmethod
    def __is_valid_ticker(__ticker: str) -> bool:
        """Checks whether ticker (__ticker) exists and is fully capitalised

        Parameters:
            @param __ticker: Contains the ticker we want to observe
            @type __ticker: str

        Return:
            @return:
                True: Ticker is given and is fully capitalised.
                False: Ticker is either not given or is not fully capitalised.
            @rtype: Boolean
        """
        if not __ticker:
            logger.error(f"Ticker symbol is missing.")
            return False
        if not __ticker.isupper():
            logger.error(f"Ticker symbol: {__ticker} should be fully capitalise.")
            return False
        return True

    def __get_ticker_details(self) -> bool:
        """
        Attempts to create a ticker object for <self.__ticker>

        Returns
            @return:
                True: Able to retrieve ticker clas.
                False: Unable to retrieve ticker details.
            @rtype: bool
        """
        __ticker_details = yf.Ticker(self.__ticker)
        if not __ticker_details:
            logger.error(f"Unable to retrieve details regarding {self.__ticker}.")
            return False
        self.__ticker_details: yf.ticker.Ticker = __ticker_details
        logger.info(self.__ticker_details)
        self.__ticker_curr_price: float = self.__ticker_details.info["regularMarketPrice"]
        return True

    def __get_expr_dates(self) -> bool:
        """
        Attempts to retrieve expiry dates for <self.__ticker>

        Returns
            @return:
                True: Able to retrieve option expiry dates for ticker.
                False: Unable to retrieve ticker option expiry dates.
            @rtype: bool
        """
        __ticker_expr_dates: Union[tuple[str, ...], None] = self.__ticker_details.options
        if not __ticker_expr_dates:
            logger.error(f"Unable to retrieve expiry dates of {self.__ticker}.")
            return False
        self.__expr_dates: tuple[str, ...] = __ticker_expr_dates
        return True

    @staticmethod
    def __is_valid_expr_date(__expiry_date: str) -> bool:
        """
        Checks whether expiry date exists.

        Parameters:
            @param __expiry_date: Contains the expiry date we wish to observe
            @type __expiry_date: str

        Return:
            @return:
                True: Expiry date is given.
                False: Expiry date not given
            @rtype: bool
        """
        if not __expiry_date:
            logger.error(f"Expiry date is missing.")
            return False
        return True

    def __is_within_margin_date(self) -> bool:
        """
        Attempts to compare the dates between today and expiry date of the option.

        Returns
            @return:
                True: Option's expiry date is within a day threshold.
                False: Option's expiry date is not within a day threshold.
            @rtype: Bool
        """
        __days_to_expr: int = (self.__expiry_date - self.__today_date).days
        if __days_to_expr < self.__DAYS_MIN or __days_to_expr > self.__DAYS_MAX:
            logger.info(f"{self.__expiry_date} is not within margin threshold.")
            return False
        self.__days_to_expr: int = __days_to_expr
        return True

    def __get_option_chain(self) -> bool:
        """
        Attempts to retrieve the option chain of all the products that have the same expiry date.

        Returns
           @return:
               True: Able to retrieve the option chain of all the products that have the common expiry date.
               False: Unable to retrieve the option-chain.
           @rtype: Boolean
        """
        __expiry_date: str = self.__expiry_date.strftime("%Y-%m-%d")  # Covert back to <str> type
        __option_chain: dataclass() = self.__ticker_details.option_chain(__expiry_date)
        if not __option_chain:
            logger.error(f"Unable to retrieve option chain from {self.__expiry_date}")
            return False
        self.__option_chain: dataclass() = __option_chain
        return True

    def __get_call_option(self) -> bool:
        """
        Attempts to retrieve the call option of the option chain.

        Returns
           @return:
               True: Able to retrieve the call option of the option chain.
               False: Unable to retrieve call option
           @rtype: bool
        """
        __calls: pd.DataFrame = self.__option_chain.calls
        if __calls is None or __calls.empty:
            logger.error(f"Unable to extract call options for expiry {self.__expiry_date}")
            return False
        self.__call_option_df: pd.DataFrame = pd.concat([__calls], ignore_index=True)
        return True

    def __get_put_option(self) -> bool:
        """
        Attempts to retrieve the put option of the option chain.

        Returns
           @return:
               True: Able to retrieve the put option of the option chain.
               False: Unable to retrieve put option
           @rtype: bool
        """
        __puts: pd.DataFrame = self.__option_chain.puts
        if __puts is None or __puts.empty:
            logger.error(f"Unable to extract puts options for expiry {self.__expiry_date}")
            return False
        self.__put_option_df: pd.DataFrame = pd.concat([__puts], ignore_index=True)
        return True

    def __filter_call_df(self) -> bool:
        """
        Attempts to filter the original call option dataFrame to only have the column 'strike' and 'impliedVolatility'

        Returns
          @return:
              True: Able to filter the original call option dataFrame
              False: Unable to filter the original call option dataFrame
          @rtype: bool
        """
        if 'strike' not in self.__call_option_df:
            logger.error(f"'strike' column missing in call option's dataframe.")
            return False

        if 'impliedVolatility' not in self.__call_option_df:
            logger.error(f"'impliedVolatility' column missing in call option's dataframe.")
            return False

        self.__clean_call_df: pd.DataFrame = self.__call_option_df[['strike', 'impliedVolatility']]
        return True

    def __filter_put_df(self) -> bool:
        """
        Attempts to filter the original put option dataFrame to only have the column 'strike' and 'impliedVolatility'

        Returns
          @return:
              True: Able to filter the original put option dataFrame
              False: Unable to filter the original put option dataFrame
          @rtype: bool
        """
        if 'strike' not in self.__put_option_df:
            logger.error(f"'strike' column missing in put option's dataframe.")
            return False

        if 'impliedVolatility' not in self.__put_option_df:
            logger.error(f"'impliedVolatility' column missing in put option's dataframe.")
            return False

        self.__clean_put_df: pd.DataFrame = self.__put_option_df[['strike', 'impliedVolatility']]
        return True

    def __filter_call_s_price(self) -> None:
        """
        Checks whether the strike price is within a specific predefined margin.
        """
        __count: int = 0
        __min_price: float = self.__PRICE_MIN_PERCENT * self.__ticker_curr_price
        __max_price: float = self.__PRICE_MAX_PERCENT * self.__ticker_curr_price
        for _, row in self.__clean_call_df.iterrows():

            if not (__min_price <= row["strike"] <= __max_price):
                continue

            __count += 1
            __idv_option_instance = IdvOption(
                implied_vol=row["impliedVolatility"],
                strike_price=row["strike"],
                days_to_expr=self.__days_to_expr,
            )
            self.__appr_call_option.append(__idv_option_instance)
        logger.success(f"Added {__count} call options")

    def __filter_put_s_price(self) -> None:
        """
        Checks whether the strike price is within a specific predefined margin.
        """
        __count: int = 0
        __min_price: float = self.__PRICE_MIN_PERCENT * self.__ticker_curr_price
        __max_price: float = self.__PRICE_MAX_PERCENT * self.__ticker_curr_price
        for _, row in self.__clean_put_df.iterrows():
            if not (__min_price <= row["strike"] <= __max_price):
                continue
            __count += 1
            __idv_option_instance = IdvOption(
                implied_vol=row["impliedVolatility"],
                strike_price=row["strike"],
                days_to_expr=self.__days_to_expr,
            )
            self.__appr_put_option.append(__idv_option_instance)
        logger.success(f"Added {__count} put options")

    @property
    def get_approved_options(self) -> (list[dataclass()], list[dataclass()]):
        """
        Returns the approved call and put options.
        Return:
            @return:
                self.__appr_call_option: Contains a list of all the individual call options approved within boundaries.
                self.__appr_put_option: Contains a list of all the individual call options approved within boundaries.
            @rtype:
                sef.__appr_call_option: list[dataclass()]
                sef.__appr_put_option: list[dataclass()]
        """
        return self.__appr_call_option, self.__appr_put_option


class BuildGraph:
    def __init__(self, __ticker: Union[str, None], __appr_call_options: list[dataclass()],
                 __appr_put_options: list[dataclass()]):
        """
        Attributes:
            self.__appr_call_option - A list that contains all the approved options in a dataclass
            self.__appr_put_option - A list that contains all the approved options in a dataclass
            self.__implied_vol - A <numpy.array> that contains only the option's implied volatility
            self.__strike_price - A <numpy.array> that contains only the option's strike price
            self.__days_to_expr - A <numpy.array> that contains only the option's days to expiry
            self.__strike_space - A <numpy.linspace> that contains a linear array of the parameters
            self.__strike_grid - A 2D array that corresponds to all possible combinations of the input array
            self.__expr_grid - A 2D array that corresponds to all possible combinations of the input array
            self.__implied_vol_grid - A 2D array that corresponds to all possible combinations of the input array
        """

        self.__appr_call_options: list[dataclass()] = []
        self.__appr_put_options: list[dataclass()] = []
        self.__ticker_price: float = 0.0
        self.__vol_curve_figure = None
        self.__vol_curve_ax = None
        self.__implied_vol: np.array = None
        self.__strike_price: np.array = None
        self.__days_to_expr: np.array = None
        self.__strike_space: np.linspace = []
        self.__strike_grid: list = []
        self.__expr_grid: list = []
        self.__implied_vol_grid: list = []

        self.__COLOR_MAP: str = "rainbow"
        self.__METHOD: str = "cubic"
        self.__PROJECTION: str = "3d"
        self.__SAMPLES: int = 100
        self.__FIGURE_SIZE: tuple[int, int] = (12, 8)
        self.__W_SPACE: float = 0.5

        if not self.__is_valid_ticker(__ticker):
            return
        self.__ticker: str = __ticker

        if not self.__is_valid_call_dataclass(__appr_call_options):
            return

        if not self.__is_valid_put_dataclass(__appr_put_options):
            return

        self.__init_vol_curve_params()
        self.__format_dataframe()
        self.__create_strike_space()
        self.__create_expr_space()
        self.__create_meshgrid()
        self.__create_implied_vol_grid()
        self.__create_graph()
        self.__add_current_price()
        self.__add_call_options()
        self.__add_put_options()
        self.__display_graph()

    @staticmethod
    def __is_valid_ticker(__ticker: str) -> bool:
        """
        Checks whether ticker (__ticker) exists and is fully capitalise
        Parameters:
                    @param __ticker: Contains the ticker we want to observe
                    @type __ticker: str

                Return:
                    @return:
                        True: Ticker is given and is fully capitalised.
                        False: Ticker is either not given or is not fully capitalised.
                    @rtype: bool
                """
        if not __ticker:
            logger.error(f"Ticker symbol is missing.")
            return False
        if not __ticker.isupper():
            logger.error(f"Ticker symbol: {__ticker} should be fully capitalise.")
            return False
        return True

    def __is_valid_call_dataclass(self, __appr_call_options: Union[list[dataclass()], None]) -> bool:
        """
        Checks whether the dataclass (containing the call option) is none.
        Parameters:
            @param __appr_call_options: Contains a list of all the dataclass of the individual call options
            @type __appr_call_options: Union[None, list[dataclass()]]
        Return:
            @return
                True - __appr_call_options is not empty
                False - __appr_call_options is empty
        """
        if not __appr_call_options:
            logger.error("Call option cannot be found.")
            return False
        self.__appr_call_options: list[dataclass()] = __appr_call_options
        return True

    def __is_valid_put_dataclass(self, __appr_put_options: Union[list[dataclass()], None]) -> bool:
        """
        Checks whether the dataclass (containing the put option) is none.
        Parameters:
            @param __appr_put_options: Contains a list of all the dataclass of the individual call options
            @type __appr_put_options: Union[None, list[dataclass()]]
        Return:
            @return
                True - __appr_put_options is not empty
                False - __appr_put_options is empty
        """
        if not __appr_put_options:
            logger.error("Put option cannot be found.")
            return False
        self.__appr_put_options: list[dataclass()] = __appr_put_options
        return True

    def __init_vol_curve_params(self) -> None:
        """
        Fetches the current ticker price and uses it to create a 3D volatility surface.
        """
        self.__ticker_price: float = yf.Ticker(self.__ticker).info["regularMarketPrice"]
        __CUSTOM_LINES = [Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=6),
                          Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=6),
                          Patch(facecolor='grey', linewidth=1)]
        __CUSTOM_LABELS = ['Call options', 'Put options', f'{self.__ticker} price: ${self.__ticker_price}']

        self.__vol_curve_figure = plt.figure(figsize=self.__FIGURE_SIZE)
        self.__vol_curve_ax = self.__vol_curve_figure.add_subplot(1, 1, 1, projection=self.__PROJECTION)
        self.__vol_curve_ax.set_xlabel('Strike Price')
        self.__vol_curve_ax.set_ylabel('Days till maturity')
        self.__vol_curve_ax.set_zlabel('Implied volatility')
        self.__vol_curve_ax.set_title(f"{self.__ticker} Interpolated Volatility Surface")
        self.__vol_curve_ax.legend(__CUSTOM_LINES, __CUSTOM_LABELS, loc='best')

    def __format_dataframe(self) -> None:
        """
        Combines both lists and split the axes such that we may compute the volatility surface.
        """
        __temp_combined_list: list[dataclass()] = self.__appr_call_options + self.__appr_put_options
        self.__implied_vol: np.array = np.array([__option.implied_vol for __option in __temp_combined_list])
        self.__strike_price: np.array = np.array([__option.strike_price for __option in __temp_combined_list])
        self.__days_to_expr: np.array = np.array([__option.days_to_expr for __option in __temp_combined_list])

    def __create_strike_space(self) -> None:
        """
        Returns an evenly spaced numbers over a specific interval.

        https://numpy.org/doc/2.1/reference/generated/numpy.linspace.html
        """
        self.__strike_space: np.linspace = np.linspace(min(self.__strike_price),
                                                       max(self.__strike_price),
                                                       self.__SAMPLES,
                                                       dtype=np.float32)

    def __create_expr_space(self) -> None:
        """
        Returns an evenly spaced numbers over a specific interval.

        https://numpy.org/doc/2.1/reference/generated/numpy.linspace.html
        """
        self.__expr_space: np.linspace = np.linspace(min(self.__days_to_expr),
                                                     max(self.__days_to_expr),
                                                     self.__SAMPLES,
                                                     dtype=np.float32)

    def __create_meshgrid(self) -> None:
        """
        Creates a coordinate grid from self.__strike_space and self.__expr_space
        """
        self.__strike_grid, self.__expr_grid = np.meshgrid(self.__strike_space, self.__expr_space)

    def __create_implied_vol_grid(self) -> None:
        """
        Create interpolated scattered data onto a grid, providing values at new coordinates based on known data points.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
        """
        self.__implied_vol_grid = griddata(
            (self.__strike_price, self.__days_to_expr), self.__implied_vol,
            (self.__strike_grid, self.__expr_grid),
            method=self.__METHOD
        )

    def __create_graph(self) -> None:
        """
        Using the X (strike_grid) and Y (expr_grid) as coordinates and Z as (implied_vol_grid) to plot the graph
        https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html
        """
        self.__vol_curve_ax.plot_surface(
            self.__strike_grid,
            self.__expr_grid,
            self.__implied_vol_grid,
            cmap=self.__COLOR_MAP
        )

    def __add_current_price(self) -> None:
        """
        Calculates and plots the graph requires displaying the current price of the underlying as a flat plane
        """
        __x: np = np.array([self.__ticker_price, self.__ticker_price])
        __y: np = np.array([min(self.__days_to_expr), max(self.__days_to_expr)])

        __z_min = np.nanmin(self.__implied_vol_grid)
        __z_max = np.nanmax(self.__implied_vol_grid)
        __z: np = np.array([__z_min, __z_max])

        __curr_price_x, __curr_price_y = np.meshgrid(__x, __y)
        __curr_price_z = np.meshgrid(__z, [1, 1])[0]

        self.__vol_curve_ax.plot_surface(__curr_price_x, __curr_price_y, __curr_price_z, color='black', alpha=0.5)

    def __add_call_options(self) -> None:
        """
        Plots all the call options into the volatility surface
        """
        for __option in self.__appr_call_options:
            self.__vol_curve_ax.scatter(__option.strike_price,
                                        __option.days_to_expr,
                                        __option.implied_vol,
                                        color='red', s=25, marker='o')

    def __add_put_options(self) -> None:
        """
        Plots all the put options into the volatility surface
        """
        for __option in self.__appr_put_options:
            self.__vol_curve_ax.scatter(__option.strike_price,
                                        __option.days_to_expr,
                                        __option.implied_vol,
                                        color='blue', s=25, marker='o')

    def __display_graph(self) -> None:
        """
        Display the graph
        """
        plt.subplots_adjust(wspace=self.__W_SPACE)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    __ticker: Union[str, None] = input("Please enter a Ticker symbol: ")
    dataHandler = RetrieveData(__ticker)
    __appr_call_options, __appr_put_options = dataHandler.get_approved_options
    BuildGraph(__ticker, __appr_call_options, __appr_put_options)

import streamlit as st
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import logging
from tqdm import tqdm
import random
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import io

# Set up logging
logging.basicConfig(filename='strategy_results.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class PriceBasedCandleStrategy:
    def __init__(self, data, initial_capital=1000, ema_period=20, threshold=0.02, 
                 stop_loss_percent=2, price_threshold=0.005, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.ema_period = ema_period
        self.threshold = threshold
        self.stop_loss_percent = stop_loss_percent
        self.price_threshold = price_threshold
        self.transaction_cost = transaction_cost
        self.original_data = data
        self.original_data['EMA'] = ta.ema(self.original_data['Close'], length=ema_period)
        self.price_based_data = self.convert_to_price_based_candles(data)
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.cash = initial_capital
        self.ticker = None
        self.start_date = None
        self.end_date = None

    def convert_to_price_based_candles(self, data):
        price_candles = []
        current_candle = data.iloc[0].copy()
        last_price = current_candle['Close']

        for i in range(1, len(data)):
            row = data.iloc[i]
            price_change = abs(row['Close'] - last_price) / last_price
            if price_change >= self.price_threshold:
                current_candle['EMA'] = data['EMA'].iloc[i-1]  # Use EMA from original data
                price_candles.append(current_candle)
                current_candle = row.copy()
                last_price = row['Close']
            else:
                current_candle['High'] = max(current_candle['High'], row['High'])
                current_candle['Low'] = min(current_candle['Low'], row['Low'])
                current_candle['Close'] = row['Close']
                current_candle['Volume'] += row['Volume']

        current_candle['EMA'] = data['EMA'].iloc[-1]  # Use last EMA from original data
        price_candles.append(current_candle)
        return pd.DataFrame(price_candles)

    def calculate_deviation(self, price, ema):
        return (price - ema) / ema

    def backtest(self):
        self.equity_curve = [self.initial_capital] * len(self.price_based_data)
        for i in range(1, len(self.price_based_data)):
            if i >= self.ema_period:
                price = self.price_based_data['Close'].iloc[i]
                ema = self.price_based_data['EMA'].iloc[i]
                deviation = self.calculate_deviation(price, ema)

                if not self.positions and abs(deviation) > self.threshold:
                    if deviation < 0:
                        self.enter_trade('long', price, ema, i)
                    else:
                        self.enter_trade('short', price, ema, i)
                elif self.positions:
                    self.check_exit_condition(i)

            self.update_equity(i)

    def enter_trade(self, position_type, price, ema, index):
        position_size = self.cash * 0.2  # Use 20% of available cash for each trade
        entry_cost = position_size * self.transaction_cost
        adjusted_position_size = position_size - entry_cost
        shares = adjusted_position_size / price
        stop_loss = price * (1 - self.stop_loss_percent / 100) if position_type == 'long' else price * (1 + self.stop_loss_percent / 100)

        position = {
            'type': position_type,
            'entry_price': price,
            'stop_loss': stop_loss,
            'entry_index': index,
            'exit_index': None,
            'exit_price': None,
            'shares': shares,
            'entry_cost': entry_cost
        }
        self.positions.append(position)
        self.cash -= position_size
        self.update_equity(index)  # Update equity immediately after entering a trade

    def check_exit_condition(self, index):
        price = self.price_based_data['Close'].iloc[index]
        ema = self.price_based_data['EMA'].iloc[index]

        for position in self.positions:
            if position['exit_index'] is None:
                if (position['type'] == 'long' and price >= ema) or (position['type'] == 'short' and price <= ema):
                    self.exit_trade(position, index, price)
                elif (position['type'] == 'long' and price <= position['stop_loss']) or (position['type'] == 'short' and price >= position['stop_loss']):
                    self.exit_trade(position, index, position['stop_loss'])

    def exit_trade(self, position, index, exit_price):
        position['exit_index'] = index
        position['exit_price'] = exit_price
        exit_cost = position['shares'] * exit_price * self.transaction_cost

        if position['type'] == 'long':
            return_amount = (exit_price - position['entry_price']) * position['shares']
        else:
            return_amount = (position['entry_price'] - exit_price) * position['shares']

        self.cash += return_amount + (position['shares'] * (position['entry_price'] if position['type'] == 'short' else exit_price)) - exit_cost
        self.trades.append(position)
        self.positions.remove(position)
        self.update_equity(index)  # Update equity immediately after closing a trade

    def update_equity(self, index):
        current_price = self.price_based_data['Close'].iloc[index]
        positions_value = sum(
            position['shares'] * current_price if position['type'] == 'long'
            else position['shares'] * (2 * position['entry_price'] - current_price)  # For short positions
            for position in self.positions
        )
        total_equity = self.cash + positions_value
        self.equity_curve[index] = total_equity

    def map_trades_to_original_data(self):
        for trade in self.trades:
            trade['original_entry_index'] = self.original_data.index.get_loc(self.price_based_data.index[trade['entry_index']])
            if trade['exit_index'] is not None:
                trade['original_exit_index'] = self.original_data.index.get_loc(self.price_based_data.index[trade['exit_index']])
        
        for position in self.positions:
            position['original_entry_index'] = self.original_data.index.get_loc(self.price_based_data.index[position['entry_index']])

    def calculate_performance_metrics(self):
        equity_curve = pd.Series(self.equity_curve)
        
        if len(self.trades) == 0:
            return {
                'Total Return (%)': 0,
                'Buy and Hold Return (%)': 0,
                'Max Drawdown (%)': 0,
                'Slope of Equity Curve': 0,
                'R-squared of Equity Curve': 0,
                'Number of Trades': 0,
                'Win Rate (%)': 0,
                'Sharpe Ratio': 0
            }

        returns = equity_curve.pct_change().dropna()
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        
        drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
        max_drawdown = drawdown.max() * 100

        x = np.arange(len(equity_curve))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, equity_curve)
        r_squared = r_value ** 2

        buy_hold_return = (self.original_data['Close'].iloc[-1] - self.original_data['Close'].iloc[0]) / self.original_data['Close'].iloc[0] * 100

        # Calculate win rate
        winning_trades = sum(1 for trade in self.trades if 
                             (trade['type'] == 'long' and trade['exit_price'] > trade['entry_price']) or
                             (trade['type'] == 'short' and trade['exit_price'] < trade['entry_price']))
        win_rate = (winning_trades / len(self.trades)) * 100 if len(self.trades) > 0 else 0

        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        daily_returns = returns.mean() * 252  # Annualized return
        daily_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (daily_returns - risk_free_rate) / daily_volatility if daily_volatility != 0 else 0

        return {
            'Total Return (%)': total_return,
            'Buy and Hold Return (%)': buy_hold_return,
            'Max Drawdown (%)': max_drawdown,
            'Slope of Equity Curve': slope,
            'R-squared of Equity Curve': r_squared,
            'Number of Trades': len(self.trades),
            'Win Rate (%)': win_rate,
            'Sharpe Ratio': sharpe_ratio
        }

    def generate_backtest_history(self):
        history = []
        for i, (index, row) in enumerate(self.price_based_data.iterrows()):
            trade_action = None
            for trade in self.trades:
                if trade['entry_index'] == i:
                    trade_action = f"{'BUY' if trade['type'] == 'long' else 'SELL'} @ {trade['entry_price']:.2f}"
                    break
                elif trade['exit_index'] == i:
                    trade_action = f"{'SELL' if trade['type'] == 'long' else 'BUY'} @ {trade['exit_price']:.2f}"
                    break

            history.append({
                'Date': index,
                'Close': row['Close'],
                'EMA': row['EMA'],
                'Action': trade_action,
                'Equity': self.equity_curve[i] if i < len(self.equity_curve) else self.equity_curve[-1]
            })
        return pd.DataFrame(history)

def run_backtest_for_ticker(ticker, start_date, end_date, initial_capital, params):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"Error: No data available for {ticker} between {start_date} and {end_date}.")
        return None

    if 'Close' not in data.columns:
        print(f"Error: 'Close' price data not available for {ticker}.")
        return None

    strategy = PriceBasedCandleStrategy(data, initial_capital=initial_capital, **params)
    strategy.ticker = ticker
    strategy.start_date = start_date
    strategy.end_date = end_date
    strategy.backtest()
    strategy.map_trades_to_original_data()

    performance_metrics = strategy.calculate_performance_metrics()
    backtest_history = strategy.generate_backtest_history()

    return {
        'ticker': ticker,
        'strategy': strategy,
        'performance_metrics': performance_metrics,
        'params': params,
        'backtest_history': backtest_history
    }

def evaluate_individual(individual, ticker, start_date, end_date, initial_capital):
    params = {
        'ema_period': max(3, int(individual[0])),  # Ensure EMA period is at least 3
        'threshold': max(0.01, individual[1]),
        'stop_loss_percent': max(0.1, individual[2]),
        'price_threshold': max(0.001, individual[3]),
        'transaction_cost': max(0.0001, individual[4])
    }
    
    result = run_backtest_for_ticker(ticker, start_date, end_date, initial_capital, params)
    
    if result:
        metrics = result['performance_metrics']
        
        if metrics['Number of Trades'] == 0 or metrics['Slope of Equity Curve'] <= 1e-6 or metrics['R-squared of Equity Curve'] < 1e-6:
            return float('inf'),

        opti_score = metrics['Max Drawdown (%)'] / (metrics['Slope of Equity Curve'] * metrics['R-squared of Equity Curve'])
        return opti_score,
    
    return float('inf'),

def optimize_strategy_genetic(ticker, start_date, end_date, initial_capital, population_size=50, generations=50):
    try:
        # Check if FitnessMin and Individual classes already exist
        if 'FitnessMin' not in globals() or 'Individual' not in globals():
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Define genes with positive ranges
        toolbox.register("ema_period", random.randint, 5, 100)
        toolbox.register("threshold", random.uniform, 0.01, 1)
        toolbox.register("stop_loss_percent", random.uniform, 0.5, 5)
        toolbox.register("price_threshold", random.uniform, 0.001, 1)
        toolbox.register("transaction_cost", random.uniform, 0.0001, 0.01)

        # Define individual and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.ema_period, toolbox.threshold, toolbox.stop_loss_percent, 
                          toolbox.price_threshold, toolbox.transaction_cost), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic operators
        toolbox.register("evaluate", evaluate_individual, ticker=ticker, start_date=start_date, 
                         end_date=end_date, initial_capital=initial_capital)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[2, 0.02, 0.2, 0.005, 0.0002], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population
        pop = toolbox.population(n=population_size)

        # Run genetic algorithm
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, 
                                       stats=stats, halloffame=hof, verbose=True)

        # Get best individual
        if len(hof) > 0:
            best_individual = hof[0]
            best_params = {
                'ema_period': max(3, int(best_individual[0])),
                'threshold': max(0.01, best_individual[1]),
                'stop_loss_percent': max(0.1, best_individual[2]),
                'price_threshold': max(0.001, best_individual[3]),
                'transaction_cost': max(0.0001, best_individual[4])
            }

            best_result = run_backtest_for_ticker(ticker, start_date, end_date, initial_capital, best_params)
            if best_result:
                best_result['opti_score'] = best_individual.fitness.values[0]
                return best_result

    except Exception as e:
        print(f"An error occurred while optimizing {ticker}: {str(e)}")
    
    return None

# Streamlit app
st.title('Price-Based Candle Strategy Optimizer')

# Sidebar inputs
st.sidebar.header('Input Parameters')

# Ticker input
ticker = st.sidebar.text_input('Ticker Symbol', 'AAPL')

# Date range
end_date = st.sidebar.date_input('End Date', datetime.now())
start_date = st.sidebar.date_input('Start Date', end_date - timedelta(days=10*365))

# Convert dates to string format
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# Other parameters
initial_capital = st.sidebar.number_input('Initial Capital', value=1000, min_value=100, step=100)
population_size = st.sidebar.number_input('Population Size', value=100, min_value=10, step=10)
generations = st.sidebar.number_input('Generations', value=100, min_value=10, step=10)

# Gene range inputs
st.sidebar.header('Gene Ranges')
ema_period_min = st.sidebar.number_input('EMA Period Min', value=5, min_value=3, step=1)
ema_period_max = st.sidebar.number_input('EMA Period Max', value=100, min_value=ema_period_min+1, step=1)
threshold_min = st.sidebar.number_input('Threshold Min', value=0.01, min_value=0.001, step=0.001, format="%.3f")
threshold_max = st.sidebar.number_input('Threshold Max', value=1.0, min_value=threshold_min+0.001, step=0.001, format="%.3f")
stop_loss_percent_min = st.sidebar.number_input('Stop Loss % Min', value=0.5, min_value=0.1, step=0.1, format="%.1f")
stop_loss_percent_max = st.sidebar.number_input('Stop Loss % Max', value=5.0, min_value=stop_loss_percent_min+0.1, step=0.1, format="%.1f")
price_threshold_min = st.sidebar.number_input('Price Threshold Min', value=0.001, min_value=0.0001, step=0.0001, format="%.4f")
price_threshold_max = st.sidebar.number_input('Price Threshold Max', value=1.0, min_value=price_threshold_min+0.0001, step=0.0001, format="%.4f")
transaction_cost_min = st.sidebar.number_input('Transaction Cost Min', value=0.0001, min_value=0.0001, step=0.0001, format="%.4f")
transaction_cost_max = st.sidebar.number_input('Transaction Cost Max', value=0.01, min_value=transaction_cost_min+0.0001, step=0.0001, format="%.4f")

# Run optimization button
if st.sidebar.button('Run Optimization'):
    st.write(f"Optimizing strategy for {ticker}...")
    
    # Update toolbox with user-defined ranges
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("ema_period", random.randint, ema_period_min, ema_period_max)
    toolbox.register("threshold", random.uniform, threshold_min, threshold_max)
    toolbox.register("stop_loss_percent", random.uniform, stop_loss_percent_min, stop_loss_percent_max)
    toolbox.register("price_threshold", random.uniform, price_threshold_min, price_threshold_max)
    toolbox.register("transaction_cost", random.uniform, transaction_cost_min, transaction_cost_max)
    
    best_result = optimize_strategy_genetic(ticker, start_date, end_date, initial_capital, population_size, generations)
    
    if best_result:
        st.write("Optimization completed successfully!")
        
        # Display performance metrics
        st.header("Performance Metrics")
        metrics = best_result['performance_metrics']
        for metric, value in metrics.items():
            st.write(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
        st.write(f"Opti Score: {best_result['opti_score']:.6f}")
        
        # Display optimized parameters
        st.header("Optimized Parameters")
        for param, value in best_result['params'].items():
            st.write(f"{param}: {value:.4f}" if isinstance(value, float) else f"{param}: {value}")
        
        # Plot equity curve
        st.header("Equity Curve")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(best_result['strategy'].price_based_data.index, best_result['strategy'].equity_curve)
        ax.set_title(f'Equity Curve for {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital')
        st.pyplot(fig)
        
        # Display backtest history
        st.header("Backtest History")
        st.dataframe(best_result['backtest_history'])
        
        # Option to download backtest history as CSV
        csv = best_result['backtest_history'].to_csv(index=False)
        st.download_button(
            label="Download Backtest History as CSV",
            data=csv,
            file_name=f"{ticker}_backtest_history.csv",
            mime="text/csv"
        )
    else:
        st.write(f"No valid strategy found for {ticker}")

# Instructions
st.sidebar.markdown("""
## Instructions
1. Enter the ticker symbol for the stock you want to optimize.
2. Set the date range for backtesting.
3. Adjust the initial capital, population size, and number of generations.
4. Fine-tune the gene ranges for optimization.
5. Click 'Run Optimization' to start the process.
6. Review the results, including performance metrics, optimized parameters, equity curve, and backtest history.
7. Download the backtest history as a CSV file if desired.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit")

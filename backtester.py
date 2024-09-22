import sys
import os

# Workaround for distutils
if sys.version_info >= (3, 12):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

import streamlit as st
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from tqdm import tqdm
import random
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import io
import traceback
from streamlit.runtime.scriptrunner import add_script_run_ctx

class PriceBasedCandleStrategy:
    def __init__(self, data, initial_capital=1000, ema_period=20, threshold=0.02, 
                 stop_loss_percent=2, price_threshold=0.005):
        self.initial_capital = initial_capital
        self.ema_period = ema_period
        self.threshold = threshold
        self.stop_loss_percent = stop_loss_percent
        self.price_threshold = price_threshold
        self.transaction_cost = 0.001  # Fixed transaction cost
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

    def backtest(self, progress_bar=None):
        self.equity_curve = [self.initial_capital] * len(self.price_based_data)
        total_steps = len(self.price_based_data)
        for i in range(1, total_steps):
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
            
            if progress_bar is not None:
                progress_bar.progress((i + 1) / total_steps)

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
            profit_loss = (exit_price - position['entry_price']) * position['shares']
        else:  # short position
            profit_loss = (position['entry_price'] - exit_price) * position['shares']

        # Subtract both entry and exit transaction costs
        profit_loss -= (position['entry_cost'] + exit_cost)

        self.cash += (position['shares'] * exit_price) + profit_loss
        position['profit_loss'] = profit_loss
        self.trades.append(position)
        self.positions.remove(position)
        self.update_equity(index)

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
                'Sharpe Ratio': 0,
                'Total Profit/Loss': 0
            }

        returns = equity_curve.pct_change().dropna()
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        
        drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
        max_drawdown = drawdown.max() * 100

        x = np.arange(len(equity_curve))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, equity_curve)
        r_squared = r_value ** 2

        buy_hold_return = (self.original_data['Close'].iloc[-1] - self.original_data['Close'].iloc[0]) / self.original_data['Close'].iloc[0] * 100

        # Calculate win rate and total profit/loss
        winning_trades = 0
        total_profit_loss = 0
        for trade in self.trades:
            if trade['profit_loss'] > 0:
                winning_trades += 1
            total_profit_loss += trade['profit_loss']

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
            'Sharpe Ratio': sharpe_ratio,
            'Total Profit/Loss': total_profit_loss
        }

    def plot_trades_and_equity(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        # Plot price-based candles and EMA
        ax1.plot(self.price_based_data.index, self.price_based_data['Close'], label='Close Price', color='black', alpha=0.7)
        ax1.plot(self.price_based_data.index, self.price_based_data['EMA'], label=f'EMA({self.ema_period})', linestyle='--', color='blue')

        # Plot trades
        for trade in self.trades:
            entry_date = self.price_based_data.index[trade['entry_index']]
            entry_price = trade['entry_price']
            exit_date = self.price_based_data.index[trade['exit_index']]
            exit_price = trade['exit_price']
            
            if trade['type'] == 'long':
                ax1.plot(entry_date, entry_price, '^', markersize=10, color='g', label='Buy' if 'Buy' not in ax1.get_legend_handles_labels()[1] else "")
                ax1.plot(exit_date, exit_price, 'o', markersize=10, color='purple', label='Close Long' if 'Close Long' not in ax1.get_legend_handles_labels()[1] else "")
            elif trade['type'] == 'short':
                ax1.plot(entry_date, entry_price, 'v', markersize=10, color='r', label='Sell' if 'Sell' not in ax1.get_legend_handles_labels()[1] else "")
                ax1.plot(exit_date, exit_price, 'o', markersize=10, color='orange', label='Close Short' if 'Close Short' not in ax1.get_legend_handles_labels()[1] else "")
            
            # Draw lines connecting entry and exit points
            ax1.plot([entry_date, exit_date], [entry_price, exit_price], color='gray', linestyle='--', alpha=0.5)

        # Highlight price-based candles
        for i in range(1, len(self.price_based_data)):
            prev_close = self.price_based_data['Close'].iloc[i-1]
            current_close = self.price_based_data['Close'].iloc[i]
            color = 'g' if current_close > prev_close else 'r'
            ax1.plot([self.price_based_data.index[i-1], self.price_based_data.index[i]], 
                    [prev_close, current_close], color=color, linewidth=2, alpha=0.7)

        ax1.set_title(f'{self.ticker} - Price-Based Candles, EMA, and Trades')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot equity curve
        ax2.plot(self.price_based_data.index, self.equity_curve, label='Equity Curve', color='green')
        ax2.axhline(self.initial_capital, linestyle='--', color='red', alpha=0.6, label='Initial Capital')
        
        # Add markers for trade entries and exits on equity curve
        for trade in self.trades:
            entry_date = self.price_based_data.index[trade['entry_index']]
            exit_date = self.price_based_data.index[trade['exit_index']]
            entry_equity = self.equity_curve[trade['entry_index']]
            exit_equity = self.equity_curve[trade['exit_index']]
            
            if trade['type'] == 'long':
                ax2.plot(entry_date, entry_equity, '^', markersize=8, color='g')
                ax2.plot(exit_date, exit_equity, 'o', markersize=8, color='purple')
            elif trade['type'] == 'short':
                ax2.plot(entry_date, entry_equity, 'v', markersize=8, color='r')
                ax2.plot(exit_date, exit_equity, 'o', markersize=8, color='orange')

        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Capital')
        ax2.legend()

        plt.tight_layout()
        return fig

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
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error(f"Error: No data available for {ticker} between {start_date} and {end_date}.")
            return None

        if 'Close' not in data.columns:
            st.error(f"Error: 'Close' price data not available for {ticker}.")
            return None

        strategy = PriceBasedCandleStrategy(data, initial_capital=initial_capital, **params)
        strategy.ticker = ticker
        strategy.start_date = start_date
        strategy.end_date = end_date
        
        progress_bar = st.progress(0)
        strategy.backtest(progress_bar)
        progress_bar.empty()

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
    except Exception as e:
        st.error(f"An error occurred while running backtest for {ticker}: {str(e)}")
        st.error(traceback.format_exc())
        return None

def evaluate_individual(individual, ticker, start_date, end_date, initial_capital):
    params = {
        'ema_period': max(3, int(individual[0])),  # Ensure EMA period is at least 3
        'threshold': max(0.01, individual[1]),
        'stop_loss_percent': max(0.1, individual[2]),
        'price_threshold': max(0.001, individual[3]),
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
        if 'FitnessMin' not in globals() or 'Individual' not in globals():
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Define genes with user-specified ranges
        toolbox.register("ema_period", random.randint, ema_period_min, ema_period_max)
        toolbox.register("threshold", random.uniform, threshold_min, threshold_max)
        toolbox.register("stop_loss_percent", random.uniform, stop_loss_min, stop_loss_max)
        toolbox.register("price_threshold", random.uniform, price_threshold_min, price_threshold_max)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.ema_period, toolbox.threshold, toolbox.stop_loss_percent, 
                          toolbox.price_threshold), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate_individual, ticker=ticker, start_date=start_date, 
                         end_date=end_date, initial_capital=initial_capital)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[2, 0.02, 0.2, 0.005], indpb=0.25)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=population_size)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Create a progress bar for generations
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def show_progress(gen, progress_bar=progress_bar, progress_text=progress_text):
            progress_bar.progress((gen + 1) / generations)
            progress_text.text(f"Generation {gen + 1}/{generations}")

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, 
                                       stats=stats, halloffame=hof, verbose=False,
                                       callback=show_progress)

        progress_bar.empty()
        progress_text.empty()

        if len(hof) > 0:
            best_individual = hof[0]
            best_params = {
                'ema_period': max(3, int(best_individual[0])),
                'threshold': max(0.01, best_individual[1]),
                'stop_loss_percent': max(0.1, best_individual[2]),
                'price_threshold': max(0.001, best_individual[3]),
            }

            best_result = run_backtest_for_ticker(ticker, start_date, end_date, initial_capital, best_params)
            if best_result:
                best_result['opti_score'] = best_individual.fitness.values[0]
                return best_result

    except Exception as e:
        st.error(f"An error occurred while optimizing {ticker}: {str(e)}")
        st.error(traceback.format_exc())
    
    return None

# Streamlit app
st.title('Trading Bot Optimizer')

# Sidebar for input parameters
st.sidebar.header('Input Parameters')

tickers = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL,INTC,SPY').split(',')
end_date = st.sidebar.date_input('End Date', datetime.now())
start_date = st.sidebar.date_input('Start Date', end_date - timedelta(days=10*365))
initial_capital = st.sidebar.number_input('Initial Capital', min_value=100, value=1000)
population_size = st.sidebar.number_input('Population Size', min_value=10, value=50)
generations = st.sidebar.number_input('Generations', min_value=10, value=50)

# Gene definition
st.sidebar.header('Gene Definition')
ema_period_min = st.sidebar.number_input('EMA Period Min', min_value=3, value=5)
ema_period_max = st.sidebar.number_input('EMA Period Max', min_value=ema_period_min+1, value=100)
threshold_min = st.sidebar.number_input('Threshold Min', min_value=0.001, value=0.01, format='%f')
threshold_max = st.sidebar.number_input('Threshold Max', min_value=threshold_min+0.001, value=1.0, format='%f')
stop_loss_min = st.sidebar.number_input('Stop Loss % Min', min_value=0.1, value=0.5, format='%f')
stop_loss_max = st.sidebar.number_input('Stop Loss % Max', min_value=stop_loss_min+0.1, value=5.0, format='%f')
price_threshold_min = st.sidebar.number_input('Price Threshold Min', min_value=0.0001, value=0.001, format='%f')
price_threshold_max = st.sidebar.number_input('Price Threshold Max', min_value=price_threshold_min+0.0001, value=1.0, format='%f')

# Run optimization
if st.sidebar.button('Run Optimization'):
    optimized_results = []

    for ticker in tickers:
        st.write(f"\nOptimizing strategy for {ticker}...")
        
        # Check data availability
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data available for {ticker} between {start_date} and {end_date}. Skipping...")
            continue
        
        best_result = optimize_strategy_genetic(ticker, start_date, end_date, initial_capital, population_size, generations)
        if best_result:
            optimized_results.append(best_result)
            
            st.write("Best parameters:")
            st.write(best_result['params'])
            st.write("Performance Metrics:")
            for metric, value in best_result['performance_metrics'].items():
                if isinstance(value, (int, float)):
                    st.write(f"  {metric}: {value:.4f}")
                else:
                    st.write(f"  {metric}: {value}")
            st.write(f"  Opti Score: {best_result['opti_score']:.6f}")

            # Display backtest history
            st.write(f"\nBacktest History for {ticker}:")
            st.dataframe(best_result['backtest_history'])

            # Plot the results for each optimized strategy
            fig = best_result['strategy'].plot_trades_and_equity()
            st.pyplot(fig)
        else:
            st.warning(f"No valid strategy found for {ticker}")

    if optimized_results:
        st.write("\nBacktest Summary:")
        summary_df = pd.DataFrame([
            {
                'Ticker': result['ticker'],
                'Total Return (%)': result['performance_metrics']['Total Return (%)'],
                'Buy & Hold (%)': result['performance_metrics']['Buy and Hold Return (%)'],
                'Max Drawdown (%)': result['performance_metrics']['Max Drawdown (%)'],
                'Trades': result['performance_metrics']['Number of Trades'],
                'Win Rate (%)': result['performance_metrics']['Win Rate (%)'],
                'Sharpe Ratio': result['performance_metrics']['Sharpe Ratio'],
                'Opti Score': result['opti_score'],
                'Optimized Parameters': f"EMA: {result['params']['ema_period']}, Threshold: {result['params']['threshold']:.4f}, Stop Loss: {result['params']['stop_loss_percent']:.2f}%, Price Threshold: {result['params']['price_threshold']:.4f}"
            }
            for result in optimized_results
        ])
        st.dataframe(summary_df)

        # Plot combined equity curves
        fig, ax = plt.subplots(figsize=(16, 8))
        for result in optimized_results:
            ax.plot(result['strategy'].price_based_data.index, result['strategy'].equity_curve, label=result['ticker'])

        ax.set_title('Optimized Equity Curves for All Tickers')
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No valid strategies found for any tickers.")

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)

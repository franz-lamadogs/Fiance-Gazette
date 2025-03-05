import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os
import yfinance as yf
import requests
from io import StringIO

# Set the style for plots
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['savefig.dpi'] = 300

# Define output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the tariff events
tariff_events = [
    {"date": "2018-03-22", "description": "US announces tariffs on $60B Chinese goods"},
    {"date": "2018-04-02", "description": "China retaliates with tariffs on $3B US goods"},
    {"date": "2018-06-15", "description": "US announces 25% tariffs on $50B Chinese goods"},
    {"date": "2018-07-06", "description": "First round of 25% tariffs implemented"},
    {"date": "2018-08-23", "description": "Second round of 25% tariffs implemented"},
    {"date": "2018-09-17", "description": "US announces 10% tariffs on $200B Chinese goods"},
    {"date": "2018-09-24", "description": "10% tariffs on $200B implemented"}
]

# Define the currency pairs to analyze with CORRECT Yahoo Finance symbols
# For Yahoo Finance, USD-based pairs are formatted as USDXXX=X
currency_pairs = {
    'CNY': 'USDCNY=X',  # USD to Chinese Yuan
    'EUR': 'USDEUR=X',  # USD to Euro (Note: Usually the inverse EURUSD is more common)
    'JPY': 'USDJPY=X',  # USD to Japanese Yen
    'GBP': 'USDGBP=X',  # USD to British Pound
    'CAD': 'USDCAD=X',  # USD to Canadian Dollar
    'AUD': 'USDAUD=X',  # USD to Australian Dollar
    'MXN': 'USDMXN=X',  # USD to Mexican Peso
    'KRW': 'USDKRW=X',  # USD to South Korean Won
    'CHF': 'USDCHF=X'   # USD to Swiss Franc
}

# Alternative data source: Federal Reserve Economic Data (FRED)
fred_series = {
    'CNY': 'DEXCHUS',  # China / U.S. Foreign Exchange Rate
    'EUR': 'DEXUSEU',  # U.S. / Euro Foreign Exchange Rate (inverted in processing)
    'JPY': 'DEXJPUS',  # Japan / U.S. Foreign Exchange Rate
    'GBP': 'DEXUSUK',  # U.S. / U.K. Foreign Exchange Rate (inverted in processing)
    'CAD': 'DEXCAUS',  # Canada / U.S. Foreign Exchange Rate
    'AUD': 'DEXUSAL',  # U.S. / Australia Foreign Exchange Rate (inverted in processing)
    'MXN': 'DEXMXUS',  # Mexico / U.S. Foreign Exchange Rate
    'KRW': 'DEXKOUS',  # South Korea / U.S. Foreign Exchange Rate
    'CHF': 'DEXSZUS'   # Switzerland / U.S. Foreign Exchange Rate
}

def fetch_fred_data(start_date, end_date):
    """
    Fetch currency exchange rate data from FRED (Federal Reserve Economic Data)
    This is a reliable free alternative to Yahoo Finance for historical currency data
    """
    currency_data = {}
    
    print("Fetching FRED currency data...")
    
    # Convert dates to FRED format (YYYY-MM-DD)
    start_str = start_date
    end_str = end_date
    
    for currency, series_id in fred_series.items():
        try:
            # FRED API an API key for production use
            # Here we're using a workaround to download the CSV directly
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_str}&coed={end_str}"
            
            response = requests.get(url)
            if response.status_code == 200:
                # Parse the CSV data
                df = pd.read_csv(StringIO(response.text), parse_dates=['DATE'], index_col='DATE')
                
                # Rename the column to the currency
                df.columns = [currency]
                
                # Some FRED series are inverted (e.g., EURUSD instead of USDJPY)
                # We need to invert those to match our standard format
                if series_id in ['DEXUSEU', 'DEXUSUK', 'DEXUSAL']:
                    df[currency] = 1 / df[currency]
                
                currency_data[currency] = df[currency]
                print(f"Successfully downloaded FRED data for {currency}")
            else:
                print(f"Failed to get FRED data for {currency}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching FRED data for {currency}: {str(e)}")
    
    # Combine all series into a single DataFrame
    if currency_data:
        # Create a DataFrame with all currencies
        df = pd.DataFrame({k: v for k, v in currency_data.items()})
        
        # Fill missing values (weekends/holidays)
        df = df.fillna(method='ffill')
        
        return df
    
    print("Could not fetch FRED data. Using synthetic data.")
    return None

def fetch_yf_currency_data(start_date, end_date):
    """
    Attempt to fetch currency data from Yahoo Finance
    """
    currency_data = {}
    
    print("Fetching Yahoo Finance currency data...")
    
    for currency, ticker in currency_pairs.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"Failed to get data for {ticker}: No data returned")
                continue
            
            # Use Adjusted Close for analysis
            currency_data[currency] = data['Adj Close']
            print(f"Successfully downloaded data for {currency}")
            
        except Exception as e:
            print(f"Failed to get data for {ticker}: {str(e)}")
    
    # Combine all currency data into a DataFrame
    if currency_data:
        df = pd.DataFrame(currency_data)
        df = df.fillna(method='ffill')
        return df
    
    return None

def generate_synthetic_data(start_date, end_date, currencies):
    """Generate synthetic exchange rate data for demonstration"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Base rates for 1 USD in each currency (approximate 2018 values)
    base_rates = {
        'EUR': 0.85,  # 1 USD = 0.85 EUR
        'JPY': 110.0, # 1 USD = 110 JPY
        'GBP': 0.75,  # 1 USD = 0.75 GBP
        'CNY': 6.5,   # 1 USD = 6.5 CNY
        'AUD': 1.3,   # 1 USD = 1.3 AUD
        'CAD': 1.3,   # 1 USD = 1.3 CAD
        'CHF': 0.95,  # 1 USD = 0.95 CHF
        'MXN': 19.0,  # 1 USD = 19 MXN
        'KRW': 1100.0 # 1 USD = 1100 KRW
    }
    
    # Keep only requested currencies
    base_rates = {k: v for k, v in base_rates.items() if k in currencies}
    
    # Create impact dates from tariff events
    impact_dates = []
    for event in tariff_events:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d")
        if start <= event_date <= end:
            days_from_start = (event_date - start).days
            # Different magnitudes for different events
            impact = 0.015 if "implemented" in event["description"].lower() else 0.01
            impact_dates.append((days_from_start, impact))
    
    # Generate random walks with impacts for each currency
    np.random.seed(42)  # For reproducibility
    days = (end - start).days + 1
    data = {}
    
    for currency, base_rate in base_rates.items():
        # Create noise pattern
        noise = np.random.normal(0, 0.002, days)
        
        # Add slight trend
        if currency == 'CNY':
            # CNY weakened against USD in 2018
            trend = np.linspace(0, 0.06, days)
        elif currency in ['JPY', 'CHF']:
            # Safe haven currencies might strengthen
            trend = np.linspace(0, -0.02, days)
        else:
            # General USD strengthening trend
            trend = np.linspace(0, 0.03, days)
            
        # Create the series with cumulative noise
        series = base_rate + np.cumsum(noise) + trend
        
        # Add impacts at tariff dates
        for idx, impact in impact_dates:
            if idx < days:
                # Different currencies react differently
                if currency == 'CNY':
                    # CNY would weaken more (USD gets stronger vs CNY)
                    series[idx:] += impact * 1.5
                elif currency in ['JPY', 'CHF']:
                    # Safe haven currencies might strengthen (USD gets weaker)
                    series[idx:] -= impact * 0.7
                else:
                    series[idx:] += impact
        
        data[currency] = series
    
    return pd.DataFrame(data, index=date_range)

def create_exchange_rate_plots(df, event_dates):
    """Create visualizations of exchange rates with tariff event markers"""
    
    # Function to add event markers to a plot
    def add_event_markers(ax):
        for event in event_dates:
            date = datetime.strptime(event["date"], "%Y-%m-%d")
            if date >= df.index.min() and date <= df.index.max():
                ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
                # Add small text for the first few events to avoid overcrowding
                if event == event_dates[0] or event == event_dates[3]:
                    ax.text(date, ax.get_ylim()[1] * 0.98, event["date"], 
                            rotation=90, verticalalignment='top', fontsize=8)
    
    # Plot 1: Individual currency plots
    for currency in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the exchange rate
        df[currency].plot(ax=ax)
        
        # Add the event markers
        add_event_markers(ax)
        
        # Format the plot
        ax.set_title(f'USD to {currency} Exchange Rate (2018)')
        ax.set_ylabel(f'{currency} per USD')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis to show months
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        # Add legend for tariff events
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='r', linestyle='--', alpha=0.7)]
        ax.legend(custom_lines, ['Tariff Event'])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/usd_{currency}_exchange_rate.png')
        plt.close()
    
    # Plot 2: Normalized comparison of all currencies
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize all currencies to 100 at the starting date
    normalized_df = df.div(df.iloc[0]) * 100
    
    # Plot all normalized currencies
    for currency in normalized_df.columns:
        normalized_df[currency].plot(ax=ax, label=currency)
    
    # Add event markers
    add_event_markers(ax)
    
    # Format the plot
    ax.set_title('Normalized Exchange Rates (Start = 100)')
    ax.set_ylabel('Index Value')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(title='Currency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normalized_exchange_rates.png')
    plt.close()
    
    # Plot 3: Percent change from beginning of period
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate percent change
    pct_change_df = ((df - df.iloc[0]) / df.iloc[0]) * 100
    
    # Plot percent changes
    for currency in pct_change_df.columns:
        pct_change_df[currency].plot(ax=ax, label=currency)
    
    # Add event markers
    add_event_markers(ax)
    
    # Format the plot
    ax.set_title('Percent Change in Exchange Rates (2018)')
    ax.set_ylabel('% Change')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(title='Currency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exchange_rate_percent_change.png')
    plt.close()
    
    # Plot 4: Focus on CNY (if available)
    if 'CNY' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot CNY exchange rate
        df['CNY'].plot(ax=ax, linewidth=2, color='blue')
        
        # Add event markers
        for event in event_dates:
            date = datetime.strptime(event["date"], "%Y-%m-%d")
            if date >= df.index.min() and date <= df.index.max():
                ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
                # Add text for each event
                ax.text(date, df['CNY'].loc[date] * 1.01, 
                        event["description"].split(" ")[0],  # Just the first word to avoid overcrowding
                        rotation=90, verticalalignment='bottom', fontsize=8)
        
        # Format the plot
        ax.set_title('USD to CNY Exchange Rate During US-China Trade War (2018)')
        ax.set_ylabel('CNY per USD')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/usd_cny_tariff_focus.png')
        plt.close()

def analyze_tariff_impact(df, event_dates, window_size=10):
    """
    Analyze the impact of tariff events on exchange rates
    by comparing before and after each event
    """
    results = []
    
    for event in event_dates:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d")
        
        # Define before and after periods
        before_start = event_date - timedelta(days=window_size)
        before_end = event_date - timedelta(days=1)
        after_start = event_date
        after_end = event_date + timedelta(days=window_size)
        
        # Ensure dates are within our data range
        if before_start < df.index.min():
            before_start = df.index.min()
        if after_end > df.index.max():
            after_end = df.index.max()
        
        # Get data before and after the event
        before_data = df[(df.index >= before_start) & (df.index <= before_end)]
        after_data = df[(df.index >= after_start) & (df.index <= after_end)]
        
        # Skip if we don't have enough data
        if len(before_data) < 3 or len(after_data) < 3:
            continue
        
        # Calculate impact for each currency
        for currency in df.columns:
            before_mean = before_data[currency].mean()
            after_mean = after_data[currency].mean()
            pct_change = ((after_mean - before_mean) / before_mean) * 100
            
            # Interpret the change
            # Since these are USD/XXX rates, an increase means USD strengthened
            if pct_change > 0:
                interpretation = "USD strengthened"
            else:
                interpretation = "USD weakened"
            
            # Add to results
            results.append({
                "Event Date": event["date"],
                "Event": event["description"],
                "Currency": currency,
                "Before Mean": before_mean,
                "After Mean": after_mean,
                "Percent Change": pct_change,
                "Interpretation": interpretation,
                "Significant": abs(pct_change) > 1.0  # Arbitrary threshold
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(f'{output_dir}/currency_reactions.csv', index=False)
    
    # Create a heatmap of the impacts
    if not results_df.empty:
        # Pivot the data for the heatmap
        pivot_data = results_df.pivot(index="Currency", columns="Event Date", values="Percent Change")
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_data, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
        plt.title("Impact of Tariff Events on Currency Exchange Rates (%)")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tariff_impact_heatmap.png')
        plt.close()
    
    return results_df

def create_correlation_analysis(df):
    """Create correlation analysis of currency movements"""
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Currency Exchange Rates")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/currency_correlation_matrix.png')
    plt.close()
    
    # Save correlation table
    corr_matrix.to_csv(f'{output_dir}/currency_correlations.csv')
    
    return corr_matrix

def create_summary_statistics(df):
    """Generate summary statistics for each currency"""
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'Mean': df.mean(),
        'Std Dev': df.std(),
        'Min': df.min(),
        'Max': df.max(),
        'Start': df.iloc[0],
        'End': df.iloc[-1],
        'Change (%)': ((df.iloc[-1] - df.iloc[0]) / df.iloc[0] * 100).round(2)
    })
    
    # Save to CSV
    summary_stats.to_csv(f'{output_dir}/currency_summary_statistics.csv')
    
    return summary_stats

def create_volatility_analysis(df, event_dates):
    """Analyze volatility around tariff events"""
    
    # Calculate rolling standard deviation (20-day window) as volatility measure
    volatility = df.rolling(window=20).std()
    
    # Plot volatility for each currency
    for currency in volatility.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot the volatility
        volatility[currency].plot(ax=ax)
        
        # Add event markers
        for event in event_dates:
            date = datetime.strptime(event["date"], "%Y-%m-%d")
            if date >= volatility.index.min() and date <= volatility.index.max():
                ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
        
        # Format the plot
        ax.set_title(f'Volatility of USD/{currency} Exchange Rate (2018)')
        ax.set_ylabel('20-day Rolling Standard Deviation')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/volatility_{currency}.png')
        plt.close()
    
    # Combine all volatilities into one plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot volatility for important currencies
    for currency in ['CNY', 'EUR', 'JPY', 'GBP']:
        if currency in volatility.columns:
            volatility[currency].plot(ax=ax, label=currency)
    
    # Add event markers
    for event in event_dates:
        date = datetime.strptime(event["date"], "%Y-%m-%d")
        if date >= volatility.index.min() and date <= volatility.index.max():
            ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
    
    # Format the plot
    ax.set_title('Comparison of Exchange Rate Volatility (2018)')
    ax.set_ylabel('20-day Rolling Standard Deviation')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(title='Currency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volatility_comparison.png')
    plt.close()

def create_trade_weighted_index(df):
    """
    Create a simple trade-weighted USD index
    
    In a more advanced analysis, you would use actual trade weights
    based on US trade volumes with each country
    """
    # Simple weights based on approximate trade volumes
    weights = {
        'CNY': 0.25,  # China is the largest US trading partner
        'EUR': 0.20,
        'CAD': 0.15,  # Canada is a major US trading partner
        'MXN': 0.15,  # Mexico is a major US trading partner
        'JPY': 0.10,
        'GBP': 0.05,
        'KRW': 0.05,
        'AUD': 0.03,
        'CHF': 0.02
    }
    
    # Keep only currencies that are in our dataset
    weights = {k: v for k, v in weights.items() if k in df.columns}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Normalize exchange rates to the start date
    normalized_df = df.div(df.iloc[0])
    
    # Calculate the weighted index
    index_values = pd.Series(0, index=df.index)
    for currency, weight in weights.items():
        index_values += normalized_df[currency] * weight
    
    # Scale to start at 100
    index_values = index_values * 100
    
    # Plot the index
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the index
    index_values.plot(ax=ax, linewidth=2, color='green')
    
    # Add event markers for tariff announcements
    for event in tariff_events:
        date = datetime.strptime(event["date"], "%Y-%m-%d")
        if date >= index_values.index.min() and date <= index_values.index.max():
            ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
    
    # Format the plot
    ax.set_title('Trade-Weighted USD Index (2018)')
    ax.set_ylabel('Index Value (Start = 100)')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trade_weighted_usd_index.png')
    plt.close()
    
    # Save the index data
    index_values.to_csv(f'{output_dir}/trade_weighted_usd_index.csv')
    
    return index_values

def main():
    # Define date range for 2018
    start_date = "2018-01-01"
    end_date = "2018-12-31"
    
    # Try to fetch data from Yahoo Finance first
    currency_data = fetch_yf_currency_data(start_date, end_date)
    
    # If Yahoo Finance fails, try FRED
    if currency_data is None or currency_data.empty:
        print("Yahoo Finance data fetch failed. Trying FRED...")
        currency_data = fetch_fred_data(start_date, end_date)
    
    # If both fail, use synthetic data
    if currency_data is None or currency_data.empty:
        print("All data sources failed. Using synthetic data for demonstration.")
        currency_data = generate_synthetic_data(start_date, end_date, list(currency_pairs.keys()))
    
    # Generate all the analysis and visualizations
    create_exchange_rate_plots(currency_data, tariff_events)
    impact_analysis = analyze_tariff_impact(currency_data, tariff_events)
    create_correlation_analysis(currency_data)
    create_summary_statistics(currency_data)
    create_volatility_analysis(currency_data, tariff_events)
    create_trade_weighted_index(currency_data)
    
    print("Analysis complete. Check the generated images and CSV files in the 'output' directory.")

if __name__ == "__main__":
    main()
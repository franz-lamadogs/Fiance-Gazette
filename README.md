# Project Title: Currency Exchange Rate Analysis During Tariff Events

## Overview
This project analyzes how currency exchange rates responded to the 2018 US-China trade war tariff events. It visualizes the impact of tariff announcements and implementations on major currencies, creating comprehensive analyses of exchange rate movements, correlations, and volatility patterns.

## Features
- Tracks exchange rate responses to seven key tariff events in 2018
- Creates visualizations showing how different currencies reacted to trade tensions
- Calculates correlation matrices between major currencies
- Measures volatility spikes following tariff announcements
- Generates a trade-weighted USD index
- Produces detailed event-impact analysis for each currency

## Files
- **gazette.py**: The main Python script for analyzing currency exchange rates and the impact of tariff events. This script fetches data, performs analysis, and generates visualizations.
  
- **requirements.txt**: A text file that lists all the required libraries for the project. You can install all dependencies at once using this file.

## Data resource:
The script attempts to fetch data from multiple sources in the following order:

- Yahoo Finance API (using USDXXX=X formatted tickers)
- Federal Reserve Economic Data (FRED)
- Synthetic data generation (as a fallback)

## Output:
The script generates the following files in the output directory:

- Individual currency exchange rate charts
- Normalized comparison of all currencies
- Percent change visualization
- USD/CNY exchange rate focus chart
- Tariff impact heatmap
- Currency correlation matrix
- Volatility analysis charts
- Trade-weighted USD index


## Setup Instructions
Clone this repository to your local machine
Create a virtual environment (optional but recommended):

   ```
  Copypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

Install required dependencies:
   ```
Copypip install -r requirements.txt
   ```

Run the analysis:
   ```
Copypython gazette.py
   ```

Check the generated images and CSV files in the output directory

## Required libraries:
The project uses the following libraries:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For creating static, animated, and interactive visualizations in Python.
- **seaborn**: For statistical data visualization based on matplotlib.
- **yfinance**: For fetching financial data from Yahoo Finance.
- **requests**: For making HTTP requests to fetch data from APIs.

## Analasyed currencies
- Chinese Yuan (CNY)
- Euro (EUR)
- Japanese Yen (JPY)
- British Pound (GBP)
- Canadian Dollar (CAD)
- Australian Dollar (AUD)
- Mexican Peso (MXN)
- South Korean Won (KRW)
- Swiss Franc (CHF)

## Key Tariff Events Analyzed
- March 22, 2018: US announces tariffs on $60B Chinese goods
- April 2, 2018: China retaliates with tariffs on $3B US goods
- June 15, 2018: US announces 25% tariffs on $50B Chinese goods
- July 6, 2018: First round of 25% tariffs implemented
- August 23, 2018: Second round of 25% tariffs implemented
- September 17, 2018: US announces 10% tariffs on $200B Chinese goods
- September 24, 2018: 10% tariffs on $200B implemented

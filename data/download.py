import yfinance as yf
import os
import pandas as pd


def download_yfinance_data(symbol, frequency, start, end, extract_to):
    """
    Downloads historical OHLCV data from Yahoo Finance and saves it as a CSV.
    """
    print(f"Downloading {symbol} data from {start} to {end}...")

    try:
        data = yf.download(symbol, start=start, end=end, interval=frequency, auto_adjust=True)

        if not data.empty:
            data.columns = ["Close", "High", "Low", "Open", "Volume"]
            data.index = pd.to_datetime(data.index).astype("int64") // 10 ** 6

            os.makedirs(extract_to, exist_ok=True)
            file_path = os.path.join(extract_to, f"{symbol}.csv")
            data.to_csv(file_path)
            print(f"Saved data for {symbol} to {file_path}")
        else:
            print(f"No data found for {symbol}.")
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")


def main():
    print("Starting Yahoo Finance data download...")

    for asset in targetCurrencies:
        target_dir = os.path.join(EXTRACTION_DIR, candle_frequency)
        download_yfinance_data(asset, candle_frequency, start_date, end_date, target_dir)

    print(f"All files downloaded and saved to {EXTRACTION_DIR}.")


if __name__ == "__main__":
    EXTRACTION_DIR = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\raw"

    # List of target stocks, ETFs, or cryptocurrencies from Yahoo Finance
    targetCurrencies = ["^DJI"]

    # targetCurrencies = [
    #     "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    #     "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    #     "NKE", "PFE", "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT", "DIS"
    # ]

    # targetCurrencies = [
    #     "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN", "AVGO",
    #     "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
    #     "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    #     "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB", "FDX",
    #     "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    #     "INTC", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    #     "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    #     "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    #     "RTX", "SBUX", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN",
    #     "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM"
    # ]

    # targetCurrencies = [
    #     "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD",
    #     "DOT-USD", "MATIC-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "UNI-USD", "ATOM-USD",
    #     "XMR-USD", "ETC-USD", "XLM-USD", "ICP-USD", "FIL-USD", "HBAR-USD", "APT-USD",
    #     "VET-USD", "NEAR-USD", "ALGO-USD", "QNT-USD", "GRT-USD", "EOS-USD", "AAVE-USD",
    #     "FLOW-USD", "EGLD-USD"
    # ]

    # targetCurrencies = [
    #     "AKBNK.IS", "ALARK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    #     "GARAN.IS", "GUBRF.IS", "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "MGROS.IS", "PETKM.IS", "PGSUS.IS",
    #     "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS",
    #     "TTKOM.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS"]

    candle_frequency = "1d"

    start_date = "1995-01-01"  # Start date for historical data
    end_date = "2024-12-31"  # End date for historical data

    main()

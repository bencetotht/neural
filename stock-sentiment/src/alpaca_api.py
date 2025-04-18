import requests
import os

class AlpacaAPI:
    def __init__(self):
        self.headers = {
            'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
            'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY')
        }

    def get_top_stocks(self) -> list[str]:
        req = requests.get('https://data.alpaca.markets/v1beta1/screener/stocks/most-actives', headers=self.headers).json()
        return list(map(lambda x: x['symbol'], req['most_actives']))

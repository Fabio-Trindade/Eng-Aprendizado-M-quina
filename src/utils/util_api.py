import requests

class UtilAPI:
    @staticmethod
    def get(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError('Erro: ', response.status_code)
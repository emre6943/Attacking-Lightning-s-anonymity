import requests

CBR = None


# Retrieves current block height from API in case of fail
# it will return a relatively new block height
def getBlockHeight():
    global CBR
    if CBR is not None:
        return CBR

    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()['height']
        return CBR
    except _:
        return 681250
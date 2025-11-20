import requests

##########################################################

TOKEN = "8132883086:AAG6kzqfO52XAyB2hI4VPMWHRRsMUNOaSFk"
CHAT_ID = "182144202"

###########################################################

def send_telegramm_message(text):
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        print("Error text coding (non UTF-8):", e)
        text = "Error text coding (non UTF-8)"
        
    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()

    except requests.exceptions.Timeout:
        print("Error: request timed out")
    except requests.exceptions.ConnectionError:
        print("Connection error with Telegram server")
    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e, "Response:", getattr(e.response, "text", "no response text"))
    except Exception as e:
        print("Unknown error:", e)

##################################

###send_telegramm_message("Pythons message")

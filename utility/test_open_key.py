import openai
import requests

def check_openai_api_key(api_key):
    if api_key is None:
        return False
    elif api_key == "":
        return False
    else:
        client = openai.OpenAI(api_key=api_key)
        try:
            client.models.list()
        except openai.AuthenticationError:
            return False
        else:
            return True


def check_hug_key(api_key):
    if api_key is None:
        return False
    elif api_key == "":
        return False
    else:
        url = "https://huggingface.co/api/whoami-v2"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(url,headers = headers)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            return False
        else:
            return True

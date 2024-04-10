import openai
import requests

def check_openai_api_key(api_key):
    """
    Checks if the provided OpenAI API key is valid.

    Args:
        api_key (str): The OpenAI API key to be checked.
        
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
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
    """
    Checks if the provided Hugging Face API key is valid.

    Args:
        api_key (str): The Hugging Face API key to be checked.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
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

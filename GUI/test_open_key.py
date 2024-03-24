import openai
import requests

def check_openai_api_key(api_key):
    print("api key:", api_key)
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
    print("api key:", api_key)
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
        #response = requests.post(url, headers=headers)
        try:
            #huggingface_hub.utils.hf_rasie_for_status(response)
            response.raise_for_status()
        #except HTTPError as e:
            #raise HTTPError("sikinti var",request = e.request, response = e.response,) from e
        except requests.HTTPError:
            return False
        else:
            return True
        #data = response.json()

#test1 = check_hug_key("")
#print("test1: ", test1)

#test2 = check_hug_key()
#print("test2: ", test2)

#test3 = check_hug_key("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")
#print("test3: ", test3)

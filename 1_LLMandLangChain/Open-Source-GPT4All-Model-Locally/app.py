# import requests
# from pathlib import Path
# from tqdm import tqdm


# local_path = './models/gpt4all-lora-quantized-ggml.bin'
# Path(local_path).parent.mkdir(parents=True, exist_ok=True)

# url = 'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin'

# # send a GET request to the URL to download the file.

# response = requests.get(url, stream=True)

# # open the file in binary mode and write the contents of the response to it in chunks.

# with open(local_path, 'wb') as f:
#     for chunk in tqdm(response.iter_content(chunk_size=8192)):
#         if chunk:
#             f.write(chunk)





import requests
from pathlib import Path
from tqdm import tqdm
import time

local_path = './models/gpt4all-lora-quantized-ggml.bin'
Path(local_path).parent.mkdir(parents=True, exist_ok=True)

url = 'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin'

# Maximum number of retries
max_retries = 5

for _ in range(max_retries):
    try:
        # Send a GET request to the URL to download the file.
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad requests

        # Open the file in binary mode and write the contents of the response to it in chunks.
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading", unit="B", unit_scale=True):
                if chunk:
                    f.write(chunk)
        print("Download completed successfully!")
        break  # Exit the loop if the download is successful

    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
        time.sleep(5)  # Wait for a few seconds before retrying

    except Exception as err:
        print("Error:", err)
        time.sleep(5)  # Wait for a few seconds before retrying

else:
    print("Max retries reached. Could not download the file.")

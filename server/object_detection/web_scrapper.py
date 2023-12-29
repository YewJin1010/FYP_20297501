import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_images(url, folder_path, num_images=3):
    # Send an HTTP request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all image tags in the HTML
        img_tags = soup.find_all('img')
        
        # Download the first num_images images
        for i, img_tag in enumerate(img_tags[:num_images]):
            img_url = img_tag.get('src')
            if img_url:
                img_url = urljoin(url, img_url)
                img_data = requests.get(img_url).content
                img_name = os.path.join(folder_path, f"image_{i+1}.jpg")
                
                with open(img_name, 'wb') as img_file:
                    img_file.write(img_data)
                    print(f"Downloaded: {img_name}")
    else:
        print(f"Failed to fetch the URL. Status Code: {response.status_code}")

# Example usage
url_to_scrape = "https://www.istockphoto.com/search/2/image-film?phrase=baking+soda"
output_folder = "downloaded_images"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Download the first 3 images
download_images(url_to_scrape, output_folder, num_images=3)

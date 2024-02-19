import requests
from bs4 import BeautifulSoup
import os

def scrap(term, max_images, choice, page=1):
    headers = {'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    site = None
    if choice == "g":
        site = 'gettyimages'
    else:
        site = 'istockphoto'

    url = f'https://{site}.com/search/2/image?phrase={term}&page={page}'
    counter = 1

    # Create a folder for the images
    folder_name = f'{term}_{site}'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    while(max_images >= counter and page <= 100):
        url = url.format(page=page)
        response = requests.get(url, headers=headers)      
        soup = BeautifulSoup(response.content, 'lxml')
        
        for element in soup.find_all('picture'):
            link = element.find('img')['src'] # Get Image Url
            img = requests.get(link)
            
            # Save the image in the corresponding folder
            with open(f'{folder_name}/image{counter}.png', 'wb') as f:
                f.write(img.content)
                print(f'Image {counter} Downloaded')

            if(max_images <= counter):
                return
            counter += 1         
        page += 1

if __name__ == "__main__":
    term = input('Enter Search Term: ').strip().replace(" ", "%20") # Encode Spaces

    max_images = None
    while max_images is None:
        try:  
            #max_images = int(input('Enter Max Images To Scrap: '))
            max_images = 60
        except ValueError:
            print('Please Enter A valid Number!')

    choice = None 
    while choice is None: 
        choice = input('Scrap From gettyimages or istockphoto?  (g/i): ')[0]
        if(choice != 'g' and choice != 'i'):
            choice = None

    scrap(term, max_images, choice=choice) # Scrap Images

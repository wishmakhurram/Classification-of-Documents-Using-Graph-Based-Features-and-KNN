from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd

# Define lists to store data
products = []

# Set up Selenium driver
chrome_driver_path = 'C:/Users/hp/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe'
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Loop through multiple pages
for i in range(1, 5):
    driver.get(f"https://www.amazon.com/s?i=specialty-aps&bbn=16225010011&rh=n%3A%2116225010011%2Cn%3A10787321&ref=nav_em__nav_desktop_sa_intl_baby_and_child_care_0_2_16_2")
    content = driver.page_source
    soup = BeautifulSoup(content, 'html.parser')

    # Find all product containers
    for a in soup.findAll('div', attrs={'class': 'a-section a-spacing-base'}):
        name = a.find('span', attrs={'class': 'a-size-base-plus a-color-base a-text-normal'})
        if name is not None:  # Check if name is not None
            products.append(name.text)

# Create DataFrame from the collected data
df = pd.DataFrame({'Product Name': products})

# Save DataFrame to CSV file
df.to_csv('Fashion.csv', index=False, encoding='utf-8')

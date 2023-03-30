import requests
import csv
import pandas as pd
import json
from typing import List
from bs4 import BeautifulSoup
from IPython.display import clear_output
import datetime


# functions

def get_subpages_list(template_url: str, num_pages: int, verbose: bool = True):
    """ Creates list of links to subpages to srape"""
    subpages_list = []
    for i in range(1, num_pages + 1):
        link = template_url + str(i)
        subpages_list.append(link)

    if verbose:
        print(f'Generated: {len(subpages_list)} new subpages to scrap')

    return subpages_list



def get_offers(subpages_list: List[str], verbose: bool = True):
    """Iterating through subpages to generate list of offer's links"""

    # creating empty list to store scraped details from subpages
    offers_list = []

    error_counter = 0
    for subpage in subpages_list:
        # searching for element of webpage code which contains useful data
        try:
            page = requests.get(subpage)
            soup = BeautifulSoup(page.content, "html.parser")
            script = soup.find('script', {'id': '__NEXT_DATA__'})
            json_string = script.string
            json_dict = json.loads(json_string)

            # searching for useful data in dictionary, data for offer's subplings are stored in 'items'
            details_dict = json_dict['props']['pageProps']['data']['searchAds']['items']


            for dict in details_dict:
                # adding details for every offer sublnk
                offers_list.append(dict)

            print(f'Added {len(details_dict)} offers from subpage: {subpage} to main dictionary. Actual number of offers: {len(offers_list)}')


        except Exception as e:
            print(f'Error while processing subpage {subpage}: {e}')
            error_counter += 1
            continue
    
    if verbose:
        clear_output()
        print('-' * 30)
        print(f'Successfuly scraped: {len(offers_list)} offers\nErrors: {error_counter}')


    # generate final list of offer links
    df = pd.DataFrame(offers_list)
    df['link'] = 'https://www.otodom.pl/pl/oferta/' + df['slug']
    offers_list = df['link'].to_list()

    # save backup file
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'offers_links_backup_{date_time}.csv'
    df['link'].to_csv(f'otodom_scraper/{file_name}')

    return offers_list



def scrape_offers(offers_list: List[str]):
    """Scraping details of offers provided in list"""

    scraped_offers_list = []

    for e, link in enumerate(offers_list, 1):
        # scraping content of link
        try:
            page = requests.get(link)
            soup = BeautifulSoup(page.content, "html.parser")

            # finding needed element and extracting dictionary
            script = soup.find('script', {'id': '__NEXT_DATA__'})
            json_string = script.string
            json_dict = json.loads(json_string)

            offer_dict = json_dict['props']['pageProps']['ad']

            # creating main dictionary for iterated offer
            details_dict = offer_dict['target']
            
            currency = offer_dict['characteristics'][0]['currency']
            if currency != 'PLN':
                print('Found currency different than PLN, offer skipped')
                continue

            # changing value types to ensure creation DataFrame
            for key, value in details_dict.items():
                details_dict[key] = str(value)

            
            # adding additional details not included in 'target
            details_dict['ID'] = offer_dict['id']
            details_dict['SLUG'] = offer_dict['slug']
            details_dict['ADVERT_TYPE '] = offer_dict['advertType']
            details_dict['CREATE_DATE'] = offer_dict['createdAt']
            details_dict['ESTATE_CATEGORY'] = offer_dict['adCategory']['name']
            details_dict['TRANSACTION_TYPE'] = offer_dict['adCategory']['type']
           

       


            try:
                details_dict['latitude'] = offer_dict['location']['coordinates']['latitude']
                details_dict['longitude'] = offer_dict['location']['coordinates']['longitude']
            except:
                details_dict['latitude'] = None
                details_dict['longitude'] = None

            try:
                details_dict['street'] = offer_dict['location']['address']['street']['name']
            except:
                details_dict['street'] = None

            try:
                details_dict['subdistrict'] = offer_dict['location']['address']['subdistrict']['name']
            except: 
                details_dict['subdistrict'] = None

            try:
                details_dict['district'] = offer_dict['location']['address']['district']['name']
            except: 
                details_dict['district'] = None

            if e % 10 == 0:
                clear_output()
            
            print(f'Added offers from subpage: {e} / {len(offers_list)} ---> {(e / len(offers_list)) * 100:.2f}% : {details_dict["SLUG"]} to offers list ')
            scraped_offers_list.append(details_dict)
        except:
            print(f'Error on page: {link}')
            continue
    

    # creating dataframe and modyfing dataframe, formatting and extraction needed data     
    df_offers_details = pd.DataFrame(scraped_offers_list)

    # removeing unuseful characters from data
    def clean_list_values(x):
        if isinstance(x, str) and '[' in x and ']' in x:
            x = x.replace('[', '').replace(']', '').replace("'", "")
        return x
    
    df_offers_details = df_offers_details.applymap(clean_list_values)

    # setting order of columns, dropping not needed and dirty data
    new_order = ['ID','SLUG','MarketType','TRANSACTION_TYPE','ESTATE_CATEGORY', 'user_type',
              'Price','Area', 'Rooms_num',  'Build_year', 'Floor_no', 'Building_floors_num', 'Building_material', 
            'Building_ownership', 'Building_type','Construction_status','Extras_types', 'Equipment_types', 'Windows_type', 'Rent', 'Heating', 'Security_types',
              'Country', 'Province','Subregion', 'City', 'district', 'subdistrict',  'street', 'latitude', 'longitude',    ]

    df_offers_details = df_offers_details.reindex(columns=new_order)

    # renaming columns
    for col in df_offers_details.columns:
        df_offers_details = df_offers_details.rename(columns={col: col.upper()})

    df_offers_details.rename(columns={'MARKETTYPE': 'MARKET_TYPE'}, inplace=True)

    # save backup file

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'offers_details_{date_time}.csv'
    df_offers_details.to_csv(f'otodom_scraper/{file_name}')


    # finalising
    print('-' * 30)
    print(f'Successfully created DataFrame\nRows: {df_offers_details.shape[0]}\nColumns: {df_offers_details.shape[1]}')
    return df_offers_details

import pandas as pd
import numpy as np
import functools
import pycountry
import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression

### Following is intended as a future shared source of utility code

visegrad = {'CZ': "Czech Republic", 'HU': "Hungary", 'PL': "Poland", 'SK': "Slovak Republic"}
CEE_non_core = {'EE': 'Estonia', 'LT': 'Lithuania', 'LV': 'Latvia', 'SI': 'Slovenia'}
eu_balkan = {'RO': 'Romania', 'BG': 'Bulgaria'}
germany_plus = {'DE': 'Germany', 'AT': "Austria"}
southern = {'IT': 'Italy', 'ES': 'Spain', 'PT': 'Portugal', 'GR': 'Greece'}
west = {'NL': 'Netherlands', 'BE': 'Belgium', 'GB': 'Great Britain', 'UK': 'United Kingdom', 'FR': 'France', 'LU': 'Luxembourg', 'IR': 'Ireland'}
north = {'NOR': 'Norway', 'SE': 'Sweden', 'DK': 'Denmark', 'FI': 'Finland'}
emu = {'AT': 'Austria', 'BE': 'Belgium', 'CY': 'Cyprus', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
       'DE': 'Germany', 'GR': 'Greece', 'IR': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg',
        'MT': 'Malta', 'NL': 'Netherlands', 'PT': 'Portugal', 'SK': "Slovak Republic", "SI": "Slovenia", "ES": "Spain"}
# NOTE eu=current EU members, eu_19 - member in 2019
# NOTE: for robustness in terms of notation we sometimes include multiple variant. e.g. UK and GB
eu = {'CZ': "Czech Republic", 'HU': "Hungary", 'PL': "Poland", 'SE': 'Sweden', 'DK': 'Denmark', 'AT': 'Austria', 'BE': 'Belgium', 'CY': 'Cyprus', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
      'DE': 'Germany', 'GR': 'Greece', 'EL': 'Greece', 'IR': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'HR': 'Croatia',
      'MT': 'Malta', 'NL': 'Netherlands', 'PT': 'Portugal', 'SK': "Slovak Republic", "SI": "Slovenia", "ES": "Spain", 'RO': 'Romania', 'BG': 'Bulgaria'}
eu_19 = {'CZ': "Czech Republic", 'HU': "Hungary", 'PL': "Poland", 'SE': 'Sweden', 'DK': 'Denmark', 'AT': 'Austria', 'BE': 'Belgium', 'CY': 'Cyprus', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
      'DE': 'Germany', 'GR': 'Greece', 'EL': 'Greece', 'IR': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'HR': 'Croatia', 'GB': 'Great Britain', 'UK': 'United Kingdom',
      'MT': 'Malta', 'NL': 'Netherlands', 'PT': 'Portugal', 'SK': "Slovak Republic", "SI": "Slovenia", "ES": "Spain", 'RO': 'Romania', 'BG': 'Bulgaria'}

# NOTE: these two are defined by Piton in Fig.1
emu_core = {'AT': 'Austria', 'BE': 'Belgium', 'FI': 'Finland', 'FR': 'France', 'DE': 'Germany', 'IT': 'Italy', 'LU': 'Luxembourg', 'NL': 'Netherlands'}
emu_periphery = {'PT': 'Portugal', 'SK': "Slovak Republic", "SI": "Slovenia", "ES": "Spain", 'CY': 'Cyprus', 'EL': 'Greece', 'IR': 'Ireland'}
non_emu = {'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'DK': 'Denmark',  'EE': 'Estonia', 'HR': 'Croatia', 'HU': 'Hungary', 'LT': 'Lithuania', 'LV': 'Latvia', 'PL': 'Poland',
           'RO': 'Romania', 'SE': 'Sweden', 'UK': 'United Kingdom'}
non_emu_ex_uk = {'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'DK': 'Denmark',  'EE': 'Estonia', 'HR': 'Croatia', 'HU': 'Hungary', 'LT': 'Lithuania', 'LV': 'Latvia', 'PL': 'Poland',
           'RO': 'Romania', 'SE': 'Sweden'}

currency_country_map = {'CZK': 'CZ', 'PLN': 'PL', 'HUF': 'HU', 'DKK': 'DK', 'SEK': 'SE', 'GBP': 'UK', 'RON': 'RO', 'BGN': 'BG'} #TODO: add Switzerland, Norway

def oecd_res2df(res: dict) -> pd.DataFrame:
    "Reads as raw OECD api query response and converts into pandas df"

    base = []
    cols = ['country']
    
    print(f"Downloading OECD table: {res['structure']['name']}")
    for st in res['structure']['dimensions']['observation']:
        if st['name'] == 'Country':
            for j in st['values']:
                base.append([j['name']])
                    
        elif st['name'] == 'Time':
            cols.extend([i['id'] for i in st['values']]) # we assume they are ordered!
    

    last_row = None
    for key, val in res['dataSets'][0]['observations'].items():
        row = int(key.split(':')[0])
        col = int(key.split(':')[-1])
            
        if last_row != row:
            base[row].extend([np.nan] * (len(cols) - 1))
            
        base[row][col+1] = val[0]
                
        last_row = row


    return pd.DataFrame(np.array(base), columns=cols)


def harmonise_base_year(gr, yr: int):
    base_val = gr.loc[lambda x: x['year'] == yr, 'value_deflator'].values[0]
    gr[f'deflator_{yr}'] = gr['value_deflator'] / base_val
    return gr


@functools.lru_cache(None)
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_2

def unify_country_codes(df):
    '''Makes the country code follow Eurostat standart'''
    country_dict = {"GB":"UK", "GR":"EL", "IR":"IE"}
    for country in country_dict:
        df.loc[df['country_code'] == country, 'country_code'] = country_dict[country]
    return df


def get_regions(df, country_col):
    df['region'] = np.where(df[country_col].isin(visegrad), "visegrad", 
                            np.where(df[country_col].isin(southern), 'south',
                                     np.where(df[country_col].isin(germany_plus), 'DE_AT',
                                              np.where(df[country_col].isin(west), 'west', 
                                                       np.where(df[country_col].isin(north), 'north',
                                                                np.where(df[country_col].isin(CEE_non_core), 'CEE_non_core',
                                                                                np.where(df[country_col].isin(eu_balkan), 'balkan', 'rest')
                                                                        )
                                                               )
                                                      )
                                              )
                                    )
                            )
    return df


def get_emu_regions(df, country_col):
    df['emu_region'] = np.where(df[country_col].isin(emu_core), 'core',
                                 np.where(df[country_col].isin(emu_periphery), 'periphery', 
                                          np.where(df[country_col].isin(non_emu), 'non_emu',
                                                   'rest')
                                         )
                                )
    return df


def get_emu_regions_ex_uk(df, country_col):
    df['emu_region'] = np.where(df[country_col].isin(emu_core), 'core',
                                 np.where(df[country_col].isin(emu_periphery), 'periphery', 
                                          np.where(df[country_col].isin(non_emu_ex_uk), 'non_emu',
                                                   'rest')
                                         )
                                )
    return df



def plot_ols_trend(endog, exog, ax):
    x = sm.add_constant(exog)

    regression = sm.OLS(endog, x)
    olsres = regression.fit()
    print(olsres.params)
    print(f'ULC variable p-value: {olsres.pvalues[1]}')
    return sm.graphics.abline_plot(model_results=olsres, ax=ax)


class Eurozone:
    def __init__(self):
        self.eurozone_1999 = {
            'BE': 'Belgium',
            'DE': 'Germany',
            'ES': 'Spain',
            'FR': 'France',
            'IE': 'Ireland',
            'IT': 'Italy',
            'LU': 'Luxembourg',
            'NL': 'Netherlands',
            'AT': 'Austria',
            'PT': 'Portugal',
            'FI': 'Finland'
        }

        self.eurozone_2001 = {**self.eurozone_1999, 'GR': 'Greece'}

        self.eurozone_2007 = {**self.eurozone_2001, 'SI': 'Slovenia'}

        self.eurozone_2008 = {**self.eurozone_2007, 'CY': 'Cyprus', 'MT': 'Malta'}

        self.eurozone_2009 = {**self.eurozone_2008, 'SK': 'Slovakia'}

        self.eurozone_2011 = {**self.eurozone_2009, 'EE': 'Estonia'}

        self.eurozone_2014 = {**self.eurozone_2011, 'LT': 'Lithuania'}

        self.eurozone_2015 = {**self.eurozone_2014, 'LV': 'Latvia'}

    def get_eurozone_members(self, year):
        for y in range(year, 1998, -1):
            if hasattr(self, f'eurozone_{y}'):
                return getattr(self, f'eurozone_{y}')
        return None
    
# Both taken from Piton (2021)
tradable_sectors = ['B', 'C', 'I', 'H', 'M_N', 'J', 'K']

non_tradable_sectors = ['F', 'D_E', 'G', 'R_S']


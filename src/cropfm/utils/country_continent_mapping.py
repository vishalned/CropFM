"""
Country to continent mapping for grid processing
Based on the actual country names from USDOS/LSIB_SIMPLE/2017 dataset
"""

COUNTRY_TO_CONTINENT = {
    # Africa
    'Abyei Area': 'Africa',
    'Algeria': 'Africa',
    'Angola': 'Africa',
    'Benin': 'Africa',
    'Botswana': 'Africa',
    'Burkina Faso': 'Africa',
    'Burundi': 'Africa',
    'Cabo Verde': 'Africa',
    'Cameroon': 'Africa',
    'Central African Rep': 'Africa',
    'Chad': 'Africa',
    'Comoros': 'Africa',
    'Cote d\'Ivoire': 'Africa',
    'Dem Rep of the Congo': 'Africa',
    'Djibouti': 'Africa',
    'Egypt': 'Africa',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Ethiopia': 'Africa',
    'Gabon': 'Africa',
    'Gambia, The': 'Africa',
    'Ghana': 'Africa',
    'Guinea': 'Africa',
    'Guinea-Bissau': 'Africa',
    'Halaib Triangle': 'Africa',
    'Kenya': 'Africa',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Madagascar': 'Africa',
    'Malawi': 'Africa',
    'Mali': 'Africa',
    'Mauritania': 'Africa',
    'Mauritius': 'Africa',
    'Mayotte': 'Africa',
    'Morocco': 'Africa',
    'Mozambique': 'Africa',
    'Namibia': 'Africa',
    'Niger': 'Africa',
    'Nigeria': 'Africa',
    'Rep of the Congo': 'Africa',
    'Reunion': 'Africa',
    'Rwanda': 'Africa',
    'Sao Tome & Principe': 'Africa',
    'Senegal': 'Africa',
    'Seychelles': 'Africa',
    'Sierra Leone': 'Africa',
    'Somalia': 'Africa',
    'South Africa': 'Africa',
    'South Sudan': 'Africa',
    'Spain (Africa)': 'Africa',
    'St Helena': 'Africa',
    'Sudan': 'Africa',
    'Swaziland': 'Africa',
    'Tanzania': 'Africa',
    'Togo': 'Africa',
    'Tunisia': 'Africa',
    'Uganda': 'Africa',
    'Western Sahara': 'Africa',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa',
    
    # Asia
    'Afghanistan': 'Asia',
    'Aksai Chin': 'Asia',
    'Armenia': 'Asia',
    'Azerbaijan': 'Asia',
    'Bahrain': 'Asia',
    'Bangladesh': 'Asia',
    'Bhutan': 'Asia',
    'Brunei': 'Asia',
    'Burma': 'Asia',
    'Cambodia': 'Asia',
    'China': 'Asia',
    'Demchok Area': 'Asia',
    'Gaza Strip': 'Asia',
    'Georgia': 'Asia',
    'Hong Kong': 'Asia',
    'IN-CH Small Disputed Areas': 'Asia',
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Iran': 'Asia',
    'Iraq': 'Asia',
    'Israel': 'Asia',
    'Japan': 'Asia',
    'Jordan': 'Asia',
    'Kalapani Area': 'Asia',
    'Kazakhstan': 'Asia',
    'Korea, North': 'Asia',
    'Korea, South': 'Asia',
    'Korean Is. (UN Jurisdiction)': 'Asia',
    'Kuwait': 'Asia',
    'Kyrgyzstan': 'Asia',
    'Laos': 'Asia',
    'Lebanon': 'Asia',
    'Liancourt Rocks': 'Asia',
    'Macau': 'Asia',
    'Malaysia': 'Asia',
    'Maldives': 'Asia',
    'Mongolia': 'Asia',
    'Nepal': 'Asia',
    'Oman': 'Asia',
    'Pakistan': 'Asia',
    'Paracel Is': 'Asia',
    'Philippines': 'Asia',
    'Qatar': 'Asia',
    'Russia': 'Asia',  # Majority of landmass
    'Saudi Arabia': 'Asia',
    'Senkakus': 'Asia',
    'Siachen-Saltoro Area': 'Asia',
    'Sinafir & Tiran Is.': 'Asia',
    'Singapore': 'Asia',
    'Spratly Is': 'Asia',
    'Sri Lanka': 'Asia',
    'Syria': 'Asia',
    'Taiwan': 'Asia',
    'Tajikistan': 'Asia',
    'Thailand': 'Asia',
    'Timor-Leste': 'Asia',
    'Turkey': 'Asia',  # Majority of landmass
    'Turkmenistan': 'Asia',
    'United Arab Emirates': 'Asia',
    'Uzbekistan': 'Asia',
    'Vietnam': 'Asia',
    'West Bank': 'Asia',
    'Yemen': 'Asia',
    
    # Europe
    'Akrotiri': 'Europe',
    'Albania': 'Europe',
    'Andorra': 'Europe',
    'Austria': 'Europe',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Bosnia & Herzegovina': 'Europe',
    'Bulgaria': 'Europe',
    'Croatia': 'Europe',
    'Cyprus': 'Europe',
    'Czechia': 'Europe',
    'Denmark': 'Europe',
    'Dhekelia': 'Europe',
    'Dragonja River Mouth': 'Europe',
    'Estonia': 'Europe',
    'Faroe Is': 'Europe',
    'Finland': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',
    'Gibraltar': 'Europe',
    'Greece': 'Europe',
    'Greenland': 'Europe',  # Part of Denmark
    'Guernsey': 'Europe',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'Ireland': 'Europe',
    'Isle of Man': 'Europe',
    'Italy': 'Europe',
    'Jan Mayen': 'Europe',
    'Jersey': 'Europe',
    'Kosovo': 'Europe',
    'Latvia': 'Europe',
    'Liechtenstein': 'Europe',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Macedonia': 'Europe',
    'Malta': 'Europe',
    'Moldova': 'Europe',
    'Monaco': 'Europe',
    'Montenegro': 'Europe',
    'Netherlands': 'Europe',
    'No Man\'s Land': 'Europe',  # Between Croatia/Serbia
    'Norway': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Portugal (Azores)': 'Europe',
    'Portugal (Madeira Is)': 'Europe',
    'Romania': 'Europe',
    'San Marino': 'Europe',
    'Serbia': 'Europe',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Spain': 'Europe',
    'Spain (Canary Is)': 'Europe',
    'Svalbard': 'Europe',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Ukraine': 'Europe',
    'United Kingdom': 'Europe',
    'Vatican City': 'Europe',
    
    # North America
    'Bahamas, The': 'North America',
    'Barbados': 'North America',
    'Belize': 'North America',
    'Bermuda': 'North America',
    'Canada': 'North America',
    'Costa Rica': 'North America',
    'Cuba': 'North America',
    'Dominica': 'North America',
    'Dominican Republic': 'North America',
    'El Salvador': 'North America',
    'Grenada': 'North America',
    'Guatemala': 'North America',
    'Haiti': 'North America',
    'Honduras': 'North America',
    'Jamaica': 'North America',
    'Mexico': 'North America',
    'Nicaragua': 'North America',
    'Panama': 'North America',
    'Saint Lucia': 'North America',
    'St Kitts & Nevis': 'North America',
    'St Vincent & the Grenadines': 'North America',
    'Trinidad & Tobago': 'North America',
    'United States': 'North America',
    'United States (Alaska)': 'North America',
    'United States (Hawaii)': 'North America',
    
    # North America - Territories/Dependencies
    'American Samoa': 'North America',
    'Anguilla': 'North America',
    'Antigua & Barbuda': 'North America',
    'Aruba': 'North America',
    'British Virgin Is': 'North America',
    'Cayman Is': 'North America',
    'Clipperton Island': 'North America',
    'Curacao': 'North America',
    'Guadeloupe': 'North America',
    'Guam': 'North America',
    'Martinique': 'North America',
    'Montserrat': 'North America',
    'Navassa I': 'North America',
    'Netherlands (Caribbean)': 'North America',
    'Northern Mariana Is': 'North America',
    'Puerto Rico': 'North America',
    'Sint Maarten': 'North America',
    'St Barthelemy': 'North America',
    'St Martin': 'North America',
    'St Pierre & Miquelon': 'North America',
    'Turks & Caicos Is': 'North America',
    'US Minor Pacific Is. Refuges': 'North America',
    'US Virgin Is': 'North America',
    'Wake I': 'North America',
    
    # South America
    'Argentina': 'South America',
    'Bolivia': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Dramana-Shakatoe Area': 'South America',
    'Ecuador': 'South America',
    'Falkland Islands': 'South America',
    'French Guiana': 'South America',
    'Guyana': 'South America',
    'Invernada Area': 'South America',
    'Isla Brasilera': 'South America',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'S Georgia & S Sandwich Is': 'South America',
    'Suriname': 'South America',
    'Uruguay': 'South America',
    'Venezuela': 'South America',
    
    # Oceania
    'Australia': 'Oceania',
    'Cook Is': 'Oceania',
    'Fed States of Micronesia': 'Oceania',
    'Fiji': 'Oceania',
    'French Polynesia': 'Oceania',
    'Kiribati': 'Oceania',
    'Marshall Is': 'Oceania',
    'Nauru': 'Oceania',
    'New Caledonia': 'Oceania',
    'New Zealand': 'Oceania',
    'Niue': 'Oceania',
    'Norfolk I': 'Oceania',
    'Palau': 'Oceania',
    'Papua New Guinea': 'Oceania',
    'Pitcairn Is': 'Oceania',
    'Samoa': 'Oceania',
    'Solomon Is': 'Oceania',
    'Tokelau': 'Oceania',
    'Tonga': 'Oceania',
    'Tuvalu': 'Oceania',
    'Vanuatu': 'Oceania',
    'Wallis & Futuna': 'Oceania',
    
    # Oceania - Australian Territories
    'Ashmore & Cartier Is': 'Oceania',
    'Christmas I': 'Oceania',
    'Cocos (Keeling) Is': 'Oceania',
    'Coral Sea Is': 'Oceania',
    'Heard I & McDonald Is': 'Oceania',
    
    # Antarctica
    'Antarctica': 'Antarctica',
    'Bouvet Island': 'Antarctica',
    'French S & Antarctic Lands': 'Antarctica',
    
    # Special/Disputed Areas
    'Bir Tawil': 'Unknown',  # Disputed between Egypt and Sudan
    'British Indian Ocean Terr': 'Unknown',  # Indian Ocean territory
    'Koualou Area': 'Unknown',  # Disputed area
}

def get_continent(country_name):
    """Get continent for a country name"""
    return COUNTRY_TO_CONTINENT.get(country_name, 'Unknown')

def get_countries_by_continent():
    """Get dictionary of countries grouped by continent"""
    continents = {}
    for country, continent in COUNTRY_TO_CONTINENT.items():
        if continent not in continents:
            continents[continent] = []
        continents[continent].append(country)
    
    # Sort countries within each continent
    for continent in continents:
        continents[continent].sort()
    
    return continents

def print_continent_summary():
    """Print summary of countries by continent"""
    continents = get_countries_by_continent()
    
    print("Country distribution by continent:")
    print("=" * 40)
    
    total_countries = 0
    for continent in sorted(continents.keys()):
        count = len(continents[continent])
        total_countries += count
        print(f"{continent}: {count} countries")
    
    print(f"\nTotal: {total_countries} countries/territories")
    
    return continents

if __name__ == "__main__":
    print_continent_summary()
    
    # Test a few mappings
    print("\nTest mappings:")
    test_countries = ['France', 'China', 'Brazil', 'Nigeria', 'Australia']
    for country in test_countries:
        print(f"{country} -> {get_continent(country)}")
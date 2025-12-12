import re
import numpy as np

# only 1 if mw part of title modelwords of product
def binary_matrix_func_titleonly(MW_title, MW_value, title_mw_list):

    def clean_mw(mw: str):
        s = str(mw)
        s = re.sub(r"diagonal", "", s, flags=re.IGNORECASE)
        return s

    #sometimes diagonal accidentally added to inch, so delete that for the set and per product
    MW_title = {
        clean_mw(mw)
        for mw in MW_title
        if clean_mw(mw) != ""
    }
    MW_list = list(MW_title.union(MW_value))
    title_mw_list = [
        {
            clean_mw(mw)
            for mw in mw_list
            if clean_mw(mw) != ""
        }
        for mw_list in title_mw_list
    ]

    ###################
    #Now make first binary matrix
    num_products = len(title_mw_list) #number of products/number of elements in title_mw_list
    num_elements = len(MW_list) #total number of model_words in MW_list

    matrix = np.zeros((num_elements, num_products)) #matrix with zeros, with per row the model words and per column the products

    # Build matrix
    for p in range(num_products):
        title_mw = title_mw_list[p]

        for i, mw in enumerate(MW_list):

            #model word from value set or from titleset, in title model words of product p
            if mw in title_mw:
                matrix[i, p] = 1
                continue

    return matrix

############################################# Additional binary matrices
#Add brand binary matrix
BRAND_KEYS = {"Brand", "Brand Name"}

def normalize_brand(raw_brand: str) -> str:
    raw_brand = raw_brand.strip().lower() #delete spaces in begin and end
    parts = re.split(r'[/,&]', raw_brand)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        #normalize so no spaces allowed inside the brand
        norm = re.sub(r'[^a-z0-9]+', '', part)
        if norm:
            tokens.append(norm)
    return tokens

#Gets brand value from a product based on "Brand"", "Brand Name"
def extract_product_brand(features_map_product):
    p_brand = set()

    if not isinstance(features_map_product, dict):
        return p_brand

    for key, val in features_map_product.items():
        if key not in BRAND_KEYS: #look at keys Brand or Brand name
            continue
        if not isinstance(val, str):
            continue

        brand_tokens = normalize_brand(val) #normalise the brand value
        for bt in brand_tokens: #only if there are several tokens
            p_brand.add(f"brand_{bt}")

    return p_brand


#extract all brands from whole featuresmap col
def extract_mw_brand_column(features_series):
    MW_brand = set() #set with all brands
    brand_per_prod_list = [] #list brand per product

    for features_map in features_series:
        mw = extract_product_brand(features_map)
        brand_per_prod_list.append(mw)
        MW_brand.update(mw)

    return MW_brand, brand_per_prod_list

#Binary matrix for brand
def brand_binary_matrix(MW_brand, brand_per_prod_list):
    #order for rows
    MW_brand_list = sorted(MW_brand)
    n_brands = len(MW_brand_list)
    n_products = len(brand_per_prod_list)
    #map brands to a row index
    brand_idx = {brand: i for i, brand in enumerate(MW_brand_list)}
    #binary brand matrix, 1 if product has the brand
    M_brand = np.zeros((n_brands, n_products), dtype=int)
    for p, product_brand in enumerate(brand_per_prod_list):
        for brand in product_brand:
            i = brand_idx[brand]
            M_brand[i, p] = 1
    return M_brand, MW_brand_list


######## Binary matrix for Maximum Resolution
MR_KEYS = {"Maximum Resolution"}

def normalize_resolution(raw_val):
    if not isinstance(raw_val, str):
        return None
    s = raw_val.strip().lower() #delete spaces begin and end
    #delete commas
    s = s.replace(",", "")
    #delete native, since occurs once
    s = re.sub(r'\s*\(native\)', '', s)
    s = s.strip()
    #recognizes patterns of number x number, but also numberxnumber etc
    m = re.match(r'^(\d+)\s*x\s*(\d+)$', s)
    if not m:
        return None
    w, h = m.groups()
    return f"{w}x{h}" #returns numberxnumber without spaces

def extract_mr_from_product(features_map):
    mr_p = set()
    if not isinstance(features_map, dict):
        return mr_p
    for key, val in features_map.items():
        if key not in MR_KEYS:
            continue
        norm = normalize_resolution(val)
        if norm:
            mr_p.add(f"mr_{norm}")
    return mr_p

#Extract set of all unique max resolution elements
def extract_mw_mr_column(features_series):
    MW_mr = set()
    mr_mw_list = []
    for fm in features_series:
        mw = extract_mr_from_product(fm)
        mr_mw_list.append(mw)
        MW_mr.update(mw)
    return MW_mr, mr_mw_list

#Make binary matrix for the maximum resolutions
def mr_binary_matrix(MW_mr, mr_mw_list):
    MW_mr_list = sorted(MW_mr)
    n_mr = len(MW_mr_list)
    n_products = len(mr_mw_list)
    mr_idx = {mw: i for i, mw in enumerate(MW_mr_list)}
    M_mr = np.zeros((n_mr, n_products), dtype=int)
    for p, mw_set in enumerate(mr_mw_list):
        for mw in mw_set:
            i = mr_idx[mw]
            M_mr[i, p] = 1
    return M_mr, MW_mr_list

##################### Binary Matrix for UPC
UPC_KEYS = {"UPC"}

def normalize_upc(raw_val):
    if not isinstance(raw_val, str):
        return None
    s = raw_val.strip()
    digits = re.sub(r'\D+', '', s) #delete nonnumerical tokens
    if not digits:
        return None
    return digits

def extract_product_upc_from_features(features_map):
    p_upc = set()
    if not isinstance(features_map, dict):
        return p_upc

    for key, val in features_map.items():
        if key not in UPC_KEYS:
            continue
        norm = normalize_upc(val)
        if norm:
            p_upc.add(f"upc_{norm}")
    return p_upc

#Get the set of all unique UPC's
def extract_mw_upc_column(features_series):
    MW_upc = set()
    upc_mw_list = []

    for fm in features_series:
        mw = extract_product_upc_from_features(fm)
        upc_mw_list.append(mw)
        MW_upc.update(mw)

    return MW_upc, upc_mw_list

def upc_binary_matrix(MW_upc, upc_mw_list):
    MW_upc_list = sorted(MW_upc)
    n_upc = len(MW_upc_list)
    n_products = len(upc_mw_list)
    upc_idx = {mw: i for i, mw in enumerate(MW_upc_list)}
    M_upc = np.zeros((n_upc, n_products), dtype=int)

    for p, mw_set in enumerate(upc_mw_list):
        for mw in mw_set:
            i = upc_idx[mw]
            M_upc[i, p] = 1

    return M_upc, MW_upc_list







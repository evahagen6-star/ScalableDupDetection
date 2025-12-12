import json
import re


def data_clean_func(text):
    # everything lowercase
    text = text.lower()

    #normalize inch and herz variants
    text = text.replace('"', 'inch')
    text = text.replace('inches', 'inch')
    text = text.replace('-inch', 'inch')
    text = text.replace('hertz', 'hz')
    text = text.replace('-hz', 'hz')

    # Remove spaces or other non-alphanumeric characters before 'inch' and 'hz'
    text = re.sub(r'[^a-z0-9]+(inch)', r'\1', text)
    text = re.sub(r'[^a-z0-9]+(hz)', r'\1', text)

    return text

#Not used in the end
# def data_clean_func_ext(text):
#     # everything lowercase
#     text = text.lower()
#
#     # normalize inch and herz variants
#     text = text.replace('"', 'inch')
#     text = text.replace('inches', 'inch')
#     text = text.replace('-inch', 'inch')
#     text = text.replace('hertz', 'hz')
#     text = text.replace('-hz', 'hz')
#
#     # Remove spaces or other non-alphanumeric characters before 'inch' and 'hz'
#     text = re.sub(r'[^a-z0-9]+(inch)', r'\1', text)
#     text = re.sub(r'[^a-z0-9]+(hz)', r'\1', text)
#
#     #Make special characters a space
#     text = re.sub(r'[\/\(\)\-\[\]\:\–\—\,\&\+\|]', ' ', text)
#
#     #Delete dot if it is not a decimal
#     text = re.sub(r'(?<!\d)\.(?!\d|\w)', ' ', text)
#
#     #if there are multpile spaces, make it only one
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     return text
